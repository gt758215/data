from __future__ import absolute_import

import signal
import threading
import time
import traceback
from collections import defaultdict, namedtuple
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import gevent
import schedule as py_schedule
from croniter import CroniterBadDateError, croniter
from gevent.pool import Pool

from mlsteam.core.const import HOST_HEARTBEAT_INTEVAL_SEC
from mlsteam.core.flavor import flavor_get_by_id, flavor_get_resources
from mlsteam.core.models import RunState
from mlsteam.core.models.cluster import ClusterType
from mlsteam.core.models.pipeline import Pipeline
from mlsteam.core.models.run import RunStateGroup, RunStateList, RunType
from mlsteam.core.plan import PlanCache, check_plan_is_overtime
from mlsteam.core.tasks.helper import TaskOperator
from mlsteam.core.user import (UserUsageCache, get_user_usage,
                               groups_get_union_users,
                               routine_update_user_usage, user_get_groups)
from mlsteam.core.utils import auditlog
from mlsteam.core.utils.dbsession import commit_on_close
from mlsteam.core.utils.socket import mlsteam_socket
from mlsteam.exception.base import MLSteamException, ResourceException
from mlsteam.exception.flavor import FlavorNotFound
from mlsteam.exception.host import HostConnectError, HostNotFound
from mlsteam.exception.plan import PlanNotFound
from mlsteam.exception.resource import (OutOfGroupLimitResource,
                                        OutOfLimitResource,
                                        OverTimeError, OverUsageError)
from mlsteam.exception.task import TaskBusy, TaskNotFound
from mlsteam.log import logger
from mlsteam.resource import KubernetesWorker, worker_resources

# This constant configures how long to wait before automatically deleting completed non-persistent jobs
GPU_TEMPERATURE_THRESHOLD = 90


class Scheduler:
    """
    Coordinates execution of Jobs
    """

    def __init__(self):
        self.running = False
        self.ready = False
        self.resources_lock = threading.RLock()
        self.shutdown = gevent.event.Event()

        self.heartbeat_inteval = HOST_HEARTBEAT_INTEVAL_SEC

        self.pipeline_cron_scheduler = None
        # {pl_uuid: {"cron": cron_utc, "next": next_trigger}}
        # next_trigger is None if next_time not available within search window
        self._pl_cron_schedules = {}
        self._pl_cron_check_time = (curr_time := datetime.now(tz=timezone.utc))
        self._pl_cron_refresh_time = curr_time + timedelta(days=30)
        self._pl_cron_lock = threading.Lock()

    def resume_resources(self, uuid):
        with self.resources_lock:
            task = TaskOperator.task_get(uuid)
            host = task.host
            if not host:
                return True
            worker_resources.release_resources(uuid)
            worker_resources.resume_resources(uuid, task.resources, host)

    def release_resources(self, uuid):
        with self.resources_lock:
            worker_resources.release_resources(uuid)
            return True

    def get_who_uses_gpus(self, username):
        groups = user_get_groups(username)
        users_in_groups = groups_get_union_users(groups)
        users = set()
        for identity in users_in_groups:
            if identity == username:
                continue
            user_runs = TaskOperator.cache_query(
                username=identity,
                status=RunStateList.USE_GPU,
                gpus=True)
            if user_runs != []:
                users = {identity} | users
        return users

    def reschedule_resources(self, username, uuid, flavor_id, force=False):
        with self.resources_lock:
            # already check gpu limit at caller side
            task = TaskOperator.task_get(uuid)
            if not task:
                raise TaskNotFound(f"Task {uuid} not found", task=uuid)

            if task.status in RunStateGroup.BOOT_STEPS and not force:
                raise TaskBusy(f"Task {uuid} is starting now", task=uuid)
            if task.status in RunStateGroup.SHUTDOWN_STEPS:
                raise TaskBusy(f'Task {uuid} is stopping now', task=uuid)
            ori_host = task.host
            current_worker = worker_resources.get_ready_worker(ori_host)
            if not current_worker:
                raise HostConnectError(f"Host {ori_host} not found", host=ori_host)

            # STEP 1. Check current host's remaining resource is enough or not
            proposed_host = ori_host
            new_resource = flavor_get_resources(flavor_id)
            try:
                if not current_worker.check_tags(flavor_tags=new_resource.get('tags') or []):
                    raise Exception('flavor tag mismatch')
                new_resource = current_worker.find_resource(new_resource, exclude_allocated_tasks=[uuid])
            except Exception as e:
                logger.info('reschedule on current worker failed, {}, skip it'.format(e))
                proposed_host = None
            # STEP 2. If porposed_host is None, Check current is failed, try another host
            if proposed_host is None:
                proposed_host, new_resource = worker_resources.find_resource_iterate(new_resource)
            # STEP 3. Really Change Resources
            # postpone releasing resources (wait for Kubernetes scheduling)
            _req = {
                'resources': new_resource,
                'flavor': flavor_id,
                'signal': signal.SIGHUP,
            }
            task.update(**_req)
            task.stop_signal_set()
            return new_resource

    def start(self):
        """
        Start the Scheduler
        Returns True on success
        """
        if not self.running:
            gevent.spawn(self.main_thread)

            self.snapshot_updater = gevent.spawn(self.snapshot_tasks)
            self.k8s_updater = gevent.spawn(self.routine_k8s_update)
            self.date_checker = gevent.spawn(self.routine_date_check)
            self.pipeline_cron_scheduler = gevent.spawn(self.routine_pipeline_cron_scheduler)
            self.running = True

        return True

    def stop(self):
        """
        Stop the Scheduler
        Returns True if the shutdown was graceful
        """
        self.shutdown.set()
        logger.debug("(Shutdown) Scheduler shutdown seted")
        wait_limit = 5
        start = time.time()
        while self.running:
            if time.time() - start > wait_limit:
                return False
            gevent.sleep(0.1)
        return True

    def main_thread(self):
        """
        Monitors the jobs in current_jobs, updates their statuses,
        and puts their tasks in queues to be processed by other threads
        """
        # Check all runs status and heartbeat. If run still in the
        # running or saving state. Maybe mlsteam system crashed. Set
        # run into done state. Each crashed run still has own heartbeat.
        # Get its heartbeat datetime for write back to auditlog. Cleanup
        # heartbeat time at last

        # ready for serving
        mlsteam_socket.ready = True
        from mlsteam.core.container.image import init_image_cache
        from mlsteam.core.template import cleanup_obsoleted_templates
        init_image_cache()
        cleanup_obsoleted_templates()
        try:
            logger.info("Initializing scheduler...")
            tasks = TaskOperator.task_load_init()
            for task in tasks:
                task.restore()
        except Exception as e:
            logger.error("scheduler cannot start, %s", str(e))
            return False
        logger.info("Scheduler ready!")
        self.ready = True
        thread_pool = Pool(10)
        routine_pool = Pool(10)
        while not self.shutdown.is_set():
            last_check = time.time()
            try:
                # Iterate backwards so we can delete jobs
                # instances

                tasks = TaskOperator.tasks_list()
                for task in tasks:
                    gevent.sleep(0)  # same as greenlet.switch()
                    # Check resume
                    if task.need_resume:
                        task.need_resume = False
                        self.resume_resources(task.uuid)
                        logger.info("Resume {} resources {}".format(task.uuid, task.resources))
                    if task.done_flag:
                        if task.signal != signal.SIGHUP:
                            if task.resources:
                                self.release_resources(task.uuid)
                                task.resources = {}
                        if task.delete_flag:
                            task.update_status(RunState.DONE)
                            task.update(status=RunState.DELETE)
                        elif task.error_flag:
                            task.update(status=RunState.ERROR)
                        else:
                            task.update(status=RunState.DONE)
                        task.done_flag = False
                    if task.status == RunState.RUN:
                        # Handle dead task reported by agent
                        # refer to core.tasks.__init__.py: report_dead_tasks()
                        if task.dead:
                            # https://github.com/myelintek/MLSteam/pull/2640
                            # Case 5. Agent task is dead
                            task.update(status=RunState.STOP)
                            logger.info('task {} is reported dead by agent'.format(task.uuid))
                            task.notify_if_error()
                            continue
                        # IMAGEPULLER IMAGEPUSHER DATAPULLER IMAGEBUILDER MODELBUILDER IMAGEFILEPUSHER CVAT
                        if not task.resources.get('local'):
                            # LAB JOB WEBAPP TENSORBOARD bellow:
                            if not task.host:
                                task.update(status=RunState.DONE)
                                logger.info('task {} is RUN without host, set DONE'.format(task.uuid))
                                continue
                            worker = worker_resources.get_ready_worker(task.host)
                            if not worker:
                                # https://github.com/myelintek/MLSteam/pull/2640
                                # Case 8. Agent is offline (or network down)
                                continue  # host not join
                            if (worker._last_heartbeat.replace(tzinfo=timezone.utc) >
                                    (task._last_update or datetime.min).replace(tzinfo=timezone.utc)):
                                # https://github.com/myelintek/MLSteam/pull/2640
                                # Case 6. Agent task is absent
                                task.update(status=RunState.STOP)
                                logger.info('task {} is disappeared on agent, set STOP'.format(task.uuid))
                                continue
                    # task heartbeat is fresh
                    if task.routine_check():
                        if task.routine_thread is None:
                            task.routine_thread = routine_pool.spawn(task.run_routine_work)
                    if task.status == RunState.STOP:
                        if task.can_stop():
                            task.update(status=RunState.STOPPING)
                            thread_pool.spawn(task.stop, err=False)
                        continue
                    elif task.status == RunState.STOPPING:
                        continue
                    # if task.status in [RunState.DONE, RunState.ERROR, RunState.ABORT]:
                    #     task.gevent_thread = None
                    #     if task.resources:
                    #         self.release_resources(task.uuid)
                    #         task.resources = {}
                    if task.status == RunState.SQUASHING:
                        if task.signal == signal.SIGHUP:
                            continue
                        if task.resources:
                            self.release_resources(task.uuid)
                            task.resources = {}
                    if task.status == RunState.DELETE:
                        logger.debug("Scheduler delete task %s", task.uuid)
                        task.update(status=RunState.DELETING)
                        self.release_resources(task.uuid)
                        thread_pool.spawn(task.delete)
                    elif task.status == RunState.INIT:
                        task.dead = False
                        task.error = ''
                        task.error_flag = False
                        task.update(status=RunState.WAIT)
                        task.message = "Scheduling"
                    elif task.status == RunState.WAIT:
                        if not task.can_start():
                            continue

                        occupied_resources = {}
                        # Lab has called user_limit_check_reserved, skip it
                        if task._type in RunType.LAB.subtypes():
                            occupied_resources = task.resources
                        else:
                            try:
                                self.user_limit_check(
                                    task.run_username or task.username, task.flavor)
                            except Exception as e:  # nosec
                                if not isinstance(e, MLSteamException):
                                    e = MLSteamException(e)
                                task.message = "{}".format(e.gettext())
                                continue
                            try:
                                # (JOB case) or others
                                err = None
                                if (task.run_no_host or  # resume a run-no-host task
                                        task.can_run_no_host() and not worker_resources.list_ready_workers()):
                                    occupied_resources = {
                                        'hostname': None,
                                        'gpus': [],
                                        'cpus': {'num': 0, 'set': ''},
                                        'memory': 0,
                                        'docker_fstype': ''
                                    }
                                    task.update(run_no_host=True)
                                else:
                                    requested_resources = flavor_get_resources(task.flavor) or task.resources
                                    try:
                                        with self.resources_lock:
                                            _, occupied_resources = worker_resources.find_resource_iterate(
                                                requested_resources, task.uuid
                                            )
                                    except Exception as e:
                                        occupied_resources = {}
                                        err = str(e)
                                if not occupied_resources:
                                    task.message = "{}".format(err)
                                    continue
                            except Exception as e:
                                logger.error('scheduler reserve task %s resources failed, %s', task.uuid, str(e))
                                task.update(status=RunState.DONE)
                                continue

                        # Start
                        task.update(status=RunState.STARTING)
                        task.message = ""
                        task.gevent_thread = thread_pool.spawn(task.start, occupied_resources)

                        # for update case, task is already in queue
                        # task.call_on_done = True
                    elif task.status in [RunState.STARTING, RunState.PULL]:
                        # redo start if task starting on host shutdown
                        if task.redo:
                            task.gevent_thread = thread_pool.spawn(task.start, task.resources)
                            task.redo = False
                        if task.dead:
                            task.update(status=RunState.STOP)
                            logger.info('task {} is reported dead by agent'.format(task.uuid))
                            task.notify_if_error()
                # save running jobs every 15 seconds
                if time.time() > last_check:
                    for task in tasks:
                        # if task.call_on_done:
                        #     # need be handled in Done Status
                        #     continue
                        if task.persistent:
                            # skip persistent
                            continue
                        if task.status not in [RunState.DONE, RunState.DELETE, RunState.ABORT, RunState.ERROR]:
                            # skip not finished state
                            continue
                        if task._last_update:
                            if (datetime.now(timezone.utc) -
                                    task._last_update.replace(tzinfo=timezone.utc)).total_seconds() < 15:
                                continue
                        logger.debug("scheduler mark delete to task %s, %s", task.uuid, task._last_update)
                        task.update(status=RunState.DELETE)
                    last_check = time.time() + 15
                gevent.sleep(0.05)
            except TaskBusy:
                # ignore error and retry
                pass
            except Exception as e:
                logger.error("caught exception from main_thread, %s", str(e))
                logger.debug("caught exception from scheduler main_thread, %s", traceback.format_exc())
        auditlog.warning('SYSTEM', auditlog.SYSTEM, "system shutdown", notify=False)
        self.ready = False
        # Start Shutdown

        # Async Kill Updater
        if isinstance(self.snapshot_updater, gevent.Greenlet):
            logger.info("(Shutdown) Scheduler Stop HeartbeatUpdater")
            self.snapshot_updater.kill(block=False)
        if isinstance(self.k8s_updater, gevent.Greenlet):
            logger.info("(Shutdown) Scheduler Stop K8sUpdater")
            self.k8s_updater.kill(block=False)
        if isinstance(self.date_checker, gevent.Greenlet):
            logger.info("(Shutdown) Scheduler Stop DateChecker")
            self.date_checker.kill(block=False)
        if isinstance(self.pipeline_cron_scheduler, gevent.Greenlet):
            logger.info("(Shutdown) Scheduler Stop PipelineCronScheduler")
            self.pipeline_cron_scheduler.kill(block=False)
        try:
            from mlsteam.core.certificate import certificate_backup
            certificate_backup()
        except Exception as e:
            logger.error("Scheduler backup certificate Error: {}".format(e))

        thread_pool.kill()
        thread_pool.join()

        logger.info("(Shutdown) Scheduler stopped")

        while not self.snapshot_updater.dead:
            gevent.sleep(0.1)

        while not self.k8s_updater.dead:
            gevent.sleep(0.1)

        while not self.date_checker.dead:
            gevent.sleep(0.1)

        while not self.pipeline_cron_scheduler.dead:
            gevent.sleep(0.1)

        self.running = False

    def user_limit_check_reserved(self, username, uuid, flavor_id):
        try:
            self.user_limit_check(username, flavor_id)

            resources = worker_resources.list_resources()
            if not resources:
                raise HostNotFound("No host found, please add host first")

            requested_resources = flavor_get_resources(flavor_id)
            with self.resources_lock:
                _, resources_found = worker_resources.find_resource_iterate(requested_resources)
            return resources_found
        except ResourceException as e:
            if str(getattr(e, "device")).lower() == 'gpu':
                gpu_users = self.get_who_uses_gpus(username)
                if len(gpu_users):
                    raise OutOfGroupLimitResource("Out of group GPU limit", gusers=gpu_users) from e
            raise e

    def user_force_gpu_check(self, username, luuid):
        """
        return answer, True or None, to check whether to stop all the cpu_lab
        if gpu_lab == 1 and cpu_lab > 0:
            return True
        else:
            return False
        """
        if username == auditlog.SYSTEM:
            return False

        current_task = TaskOperator.task_get(luuid)
        if current_task.flavor_obj.get('gpu') == 0:
            return False

        labs = TaskOperator.cache_query(
            username=username,
            run_type=RunType.LAB.subtypes(),
            status=[RunState.RUN])

        gpus, cpus = 0, 0
        for lab in labs:
            if lab.flavor_obj.get('gpu', 0) != 0:
                gpus += 1
                # check gpu_lab > 1
                if gpus > 1:
                    return False
            else:
                cpus += 1

        return gpus == 1 and cpus > 0

    def user_limit_check(self, username, flavor_id, from_flavor=0):
        if flavor_id == 0:
            return  # allow system task by flavor id = 0
        if username == auditlog.SYSTEM:
            return

        from mlsteam.core.user import user_get
        cuser = user_get(username)
        plan: PlanCache = cuser.plan_obj
        usage: UserUsageCache = get_user_usage(cuser.id)
        if not plan:
            raise PlanNotFound(f"Plan {cuser.plan} not found", plan=cuser.plan)

        flavor = flavor_get_by_id(flavor_id)
        if not flavor:
            raise FlavorNotFound(f"Flavor {flavor_id} not found", flavor=flavor_id)

        flavor_2 = flavor_get_by_id(from_flavor)

        occupied = TaskOperator.task_resource_occupied(username)

        if plan.is_overtime:
            raise OverTimeError(
                f"The current time is beyond the specified time interval. "
                f"Start time: {plan.start_time.astimezone() if plan.start_time else 'is not set'}, "
                f"End time: {plan.end_time.astimezone() if plan.end_time else 'is not set'}")

        if flavor.gpu and usage and usage.is_overusage:
            raise OverUsageError(
                f"The {cuser.username}\'s usage({usage.get_usage_minutes / 60 :.1f} hrs) is beyond the "
                f"specified usage({usage.get_quota_hours} hrs)",
                plan=plan.name)

        # check resource limit
        DeviceType = namedtuple('DeviceType', ['res', 'unit'])
        devices = {
            'gpu': DeviceType('GPU', ''),
            'cpu': DeviceType('CPU', ''),
            'memory': DeviceType('memory', 'MB'),
            'cpu_lab': DeviceType('CPU lab', '')
        }
        for check_item in ['gpu', 'cpu', 'memory', 'cpu_lab']:
            check_passed = True
            res, unit = None, None
            if check_item == 'gpu':
                if flavor.gpu_type.lower() == 'any':
                    if not any((gpu_val == -1 for gpu_val, _ in plan.gpu)):
                        limit = sum((gpu_val for gpu_val, _ in plan.gpu))
                        current = occupied['gpu']['any']
                        request = getattr(flavor, 'gpu', 0) - getattr(flavor_2, 'gpu', 0)
                        if request + current > limit:
                            res = f"{devices['gpu'].res}-{flavor.gpu_type}"
                            unit = devices['gpu'].unit
                            check_passed = False
                elif flavor.gpu_type:
                    plan_gpu_dict = plan.gpu_dict
                    request = getattr(flavor, 'gpu', 0) - (
                        # We choose to be more conservative here:
                        # reduce the request count only when gpu types match exactly
                        # (if flavor_2.gpu_type == 'any', the exact allocated gpu type is uncertain)
                        getattr(flavor_2, 'gpu', 0) if getattr(flavor_2, 'gpu_type', None) == flavor.gpu_type else 0
                    )
                    if flavor.gpu_type in plan_gpu_dict:  # priority 1: specific gpu
                        limit = plan_gpu_dict[flavor.gpu_type]
                        current = occupied['gpu'].get(flavor.gpu_type, 0)
                    elif 'any' in plan_gpu_dict:  # priority 2: any gpu
                        limit = plan_gpu_dict['any']
                        current = occupied['gpu']['any']
                    else:  # gpu type not allowed by plan
                        limit = 0
                        current = occupied['gpu']['any']
                    if limit >= 0 and request + current > limit:
                        res = f"{devices['gpu'].res}-{flavor.gpu_type}"
                        unit = devices['gpu'].unit
                        check_passed = False
            else:
                limit = getattr(plan, check_item, 0)
                if limit >= 0:
                    current = occupied[check_item]
                    request = getattr(flavor, check_item, 0) - getattr(flavor_2, check_item, 0)
                    if request + current > limit:
                        res = devices[check_item].res
                        unit = devices[check_item].unit
                        check_passed = False
            if not check_passed:
                raise OutOfLimitResource(
                    f"Reached your {res} limit: current={current}{unit}, "
                    f"request={request}{unit}, limit={limit}{unit}",
                    device=res, limit=limit, current=current, request=request, unit=unit)
        # Check preserved resources
        if not plan.preserved:
            a_attempt = {
                'cpu': flavor.cpu,
                'memory': flavor.memory,
                'gpu': {self._norm_gtype(flavor.gpu_type): flavor.gpu}
            }
            if flavor_2:
                a_attempt.update({
                    'cpu': a_attempt['cpu'] - flavor_2.cpu,
                    'memory': a_attempt['memory'] - flavor_2.memory,
                    'gpu': {gtype: a_attempt['gpu'].get(gtype, 0) -
                            (flavor_2.gpu if gtype == self._norm_gtype(flavor_2.gpu_type) else 0)
                            for gtype in {self._norm_gtype(flavor.gpu_type), self._norm_gtype(flavor_2.gpu_type)}}
                })
            worker_resources.check_preserve(allocate_attempt=a_attempt, use_preserved=False)

    @classmethod
    def _norm_gtype(cls, raw_gpu_type: str) -> str:
        return 'any' if raw_gpu_type.lower() == 'any' else raw_gpu_type

    def snapshot_tasks(self):
        from mlsteam.core.consumption.task import (get_latest_tasks,
                                                   insert_task_snapshot,
                                                   insert_taskdown,
                                                   insert_taskup,
                                                   save_task_archive,
                                                   save_task_spec)
        period = 1  # sec
        task_types = \
            RunType.LAB.subtypes() + \
            RunType.JOB.subtypes() + \
            RunType.CVAT.subtypes() + \
            RunType.LABELSTUDIO.subtypes() + \
            RunType.ANNOTATION.subtypes() + \
            RunType.WEBAPP.subtypes()
        # init
        latest = get_latest_tasks()
        # loop
        while not self.shutdown.is_set():
            gevent.sleep(period)
            if not self.ready:
                continue
            running_uuid = {}
            tasks = TaskOperator.tasks_list()
            for task in tasks:
                try:
                    # filter task
                    if task._type not in task_types:
                        continue
                    if task.status not in [RunState.RUN]:
                        continue
                    # check host alive?
                    if task.uuid in latest:
                        running_uuid[task.uuid] = latest[task.uuid]
                        continue
                    # [(gpu_uuid, gpu_index)]
                    gpus = task.resources.get('gpus')
                    # {'num': 2, 'set': '0,1'}
                    cpus = task.resources.get('cpus')

                    # save task basic information
                    save_task_archive(
                        uuid=task.uuid,
                        _type=task._type,
                        project_uuid=task.project_uuid,
                        image=task.attrs.get('from_image_short') or task.image,
                        owner=task.username,
                    )
                    # save flavor basic information
                    _flavor = flavor_get_by_id(task.flavor)
                    if not _flavor:
                        continue  # NoneFlavor.__bool__ is False
                    spec_id = save_task_spec(
                        cpu=_flavor.cpu,
                        memory=_flavor.memory,
                        gpu=_flavor.gpu,
                        gpu_type=_flavor.gpu_type,
                    )
                    # put and use later
                    running_uuid[task.uuid] = {
                        'uuid': task.uuid,
                        'username': task.run_username,
                        'spec_id': spec_id,
                        'hostname': task.host,
                        'gpus': gpus,
                        'cpus': cpus,
                    }
                except Exception as e:
                    logger.error(str(e))
            from mlsteam.core.tasks.base import TaskBase
            for _uuid in running_uuid:
                task: TaskBase = TaskOperator.task_get(_uuid)

                _flavor = flavor_get_by_id(task.flavor)
                is_overusage = routine_update_user_usage(
                    user_id=task.user_id,
                    tuuid=_uuid,
                    start_time=task.start_time
                )

                user_usage = get_user_usage(task.user_id)
                if _flavor.gpu and is_overusage:
                    auditlog.info(
                        task.username,
                        auditlog.SYSTEM,
                        f'The task with Name {task.name} has been closed by the system because '
                        f'the user\'s usage({user_usage.get_usage_minutes / 60 :.1f} hrs) has exceeded '
                        f'the limit({user_usage.get_quota_hours} hrs).',
                        task.name,
                        project=task.project_uuid
                    )
                    TaskOperator.task_stop(task, force=False)

            if set(latest) == set(running_uuid):
                continue
            # print(latest, running_uuid)
            new_tasks = set(running_uuid) - set(latest)
            absent_tasks = set(latest) - set(running_uuid)
            # remove absent task
            current = {_uuid: latest[_uuid] for _uuid in running_uuid if _uuid in latest}
            # insert TaskUp for new tasks
            for _uuid in new_tasks:
                try:
                    taskup_id = insert_taskup(**running_uuid[_uuid])
                    current[_uuid] = taskup_id
                    logger.info('****new TaskUp.id={}'.format(taskup_id))
                except Exception as e:
                    logger.error(str(e))
            for _uuid in absent_tasks:
                insert_taskdown(uuid=_uuid, taskup_id=latest[_uuid])
            # insert task list snapshot
            insert_task_snapshot(current.values())
            logger.info('****snapshot={}'.format(sorted(current.items())))
            latest = current

    class K8sWatcher:
        def __init__(self, cid, k8s_cluster):
            from mlsteam.core.cluster import KubernetesCluster
            self.cid = cid
            self.cluster: KubernetesCluster = k8s_cluster
            self.stop_event = threading.Event()
            self.pod_watcher = None
            self.pod_events = []
            self.pod_lock = threading.RLock()

        def run(self):
            self.pod_watcher = gevent.spawn(self._start_pod_watcher)

        def stop(self):
            self.stop_event.set()

        def join(self, timeout):
            gevent.joinall([self.pod_watcher], timeout=timeout)

        def _start_pod_watcher(self):
            from mlsteam.core.kubernetes import KubernetesTool
            watcher = None
            while not self.stop_event.is_set():
                try:
                    if not watcher:
                        watcher = self.cluster.k8s_tool.list_pods(namespace=None, watch=True)
                    event = next(watcher)
                    if event[0] != KubernetesTool.WatchEvent.NOOP:
                        with self.pod_lock:
                            self.pod_events.append(event)
                            logger.debug('pod_event: %s %s', event[0].name,
                                         f"{len(event[1]['items'])} pods" if 'items' in event[1]
                                         else event[1]['metadata']['name'])
                except Exception:
                    with suppress(Exception):
                        watcher.close()
                    watcher = None
                    time.sleep(10)  # avoid intense connection to (temporarily) failed clusters
                time.sleep(0.1)

            with suppress(Exception):
                watcher.close()

    def routine_k8s_update(self):
        from mlsteam.core.cluster import KubernetesCluster, KubernetesTool, cluster_manager

        @dataclass
        class PodEvent:
            pod: KubernetesWorker.PodInfo

            def __post_init__(self):
                self.pod_worker: KubernetesWorker = (worker_resources.get_worker(self.pod.hostname)
                                                     if self.pod.hostname else None)

        def get_pods(pod_items) -> dict[str, list[KubernetesWorker.PodInfo]]:
            node_pods = defaultdict(list)
            for pod_dict in pod_items:
                pod = KubernetesWorker.PodInfo.from_dict(pod_dict)
                if pod.hostname:
                    node_pods[pod.hostname].append(pod)
            return node_pods

        def update_pod_status(tuuid, pinfo: KubernetesWorker.PodInfo, deleted=False):
            if (task := TaskOperator.task_get(tuuid)):
                task.update_pod_status(pinfo, deleted=deleted)

        k8s_watcher = {}
        last_node_update, last_log_error, suppressed_cnt = {}, {}, defaultdict(int)
        while not self.shutdown.is_set():
            for cluster in cluster_manager.clusters.values():
                cid = cluster.cluster_id
                if cluster.cluster_type == ClusterType.KUBERNETES:
                    cluster: KubernetesCluster
                    try:
                        # update nodes
                        curr_time = time.time()
                        node_wait_second = curr_time - last_node_update.get(cid, 0)
                        if (not cluster.all_node_ready and node_wait_second >= 5) or node_wait_second >= 30:
                            cluster.setup_nodes()
                            last_node_update[cid] = curr_time
                        # update pods
                        if not (watcher := k8s_watcher.get(cid)):
                            watcher = self.K8sWatcher(cid, cluster)
                            watcher.run()
                            k8s_watcher[cid] = watcher
                        with watcher.pod_lock, self.resources_lock:
                            processed_events = 0
                            try:
                                for event in watcher.pod_events:
                                    if processed_events and time.time() - curr_time >= 3:
                                        break
                                    match event[0]:
                                        case KubernetesTool.WatchEvent.ADDED:
                                            pod_event = PodEvent(KubernetesWorker.PodInfo.from_dict(event[1]))
                                            if pod_event.pod_worker:
                                                pod_event.pod_worker.add_pod(pod_event.pod)
                                        case KubernetesTool.WatchEvent.DELETED:
                                            pod_event = PodEvent(KubernetesWorker.PodInfo.from_dict(event[1]))
                                            if pod_event.pod_worker:
                                                pod_event.pod_worker.del_pod(pod_event.pod)
                                            if (task_uuid := pod_event.pod.task_uuid):
                                                gevent.spawn(update_pod_status, task_uuid, pod_event.pod, deleted=True)
                                        case KubernetesTool.WatchEvent.MODIFIED:
                                            pod_event = PodEvent(KubernetesWorker.PodInfo.from_dict(event[1]))
                                            if pod_event.pod_worker:
                                                pod_event.pod_worker.update_pod(pod_event.pod)
                                            if not pod_event.pod.pod_ready and (task_uuid := pod_event.pod.task_uuid):
                                                gevent.spawn(update_pod_status, task_uuid, pod_event.pod)
                                        case KubernetesTool.WatchEvent.RENEW:
                                            node_pods = get_pods(event[1]['items'])
                                            for pod_hostname in node_pods.keys():
                                                worker: KubernetesWorker = worker_resources.get_worker(pod_hostname)
                                                if not worker:
                                                    logger.debug("cannot process event due to worker not found: {}".format(pod_hostname))
                                                    continue
                                                deleted_pods, unready_pods = worker.renew_pods(node_pods[pod_hostname])
                                                for pod in deleted_pods:
                                                    if (task_uuid := pod.task_uuid):
                                                        gevent.spawn(update_pod_status, task_uuid, pod, deleted=True)
                                                for pod in unready_pods:
                                                    if (task_uuid := pod.task_uuid):
                                                        gevent.spawn(update_pod_status, task_uuid, pod)
                                    processed_events += 1
                            finally:
                                if processed_events:
                                    del watcher.pod_events[:processed_events]
                    except Exception:
                        if curr_time - last_log_error.get(cid, 0) >= 180:
                            logger.exception('Kubernetes update error (%d errors suppressed)', suppressed_cnt[cid])
                            last_log_error[cid] = curr_time
                            suppressed_cnt[cid] = 0
                        else:
                            suppressed_cnt[cid] = suppressed_cnt[cid] + 1
                elif cid in k8s_watcher:
                    # cluster removed
                    with suppress(Exception):
                        k8s_watcher[cid].stop()
                    del k8s_watcher[cid]
            time.sleep(1)
        for watcher in k8s_watcher.values():
            watcher.stop()
        for watcher in k8s_watcher.values():
            watcher.join(timeout=30)

    def routine_date_check(self):

        py_schedule.every(1).minute.do(check_plan_is_overtime)

        while not self.shutdown.is_set():
            py_schedule.run_pending()
            time.sleep(1)

    def routine_pipeline_cron_scheduler(self):

        from mlsteam.utils.feature_flag import feature_config_get
        self.update_pipeline_cron_scheduler()

        while not self.shutdown.is_set():
            pipeline_support = feature_config_get('pipeline_support')
            if not pipeline_support:
                time.sleep(5)
                continue
            self._pl_cron_check_time = datetime.now(tz=timezone.utc)

            with self._pl_cron_lock:
                if self._pl_cron_check_time >= self._pl_cron_refresh_time:
                    # refresh the missing trigger times (previously unavailable within search window)
                    logger.info('(Scheduler) Start refreshing pipeline cron schedules')
                    for _uuid, _schedule in self._pl_cron_schedules.items():
                        if _schedule['next'] is None:
                            _schedule['next'] = self._get_pl_cron_next(_schedule['cron'], _uuid)
                    self._pl_cron_refresh_time = self._pl_cron_check_time + timedelta(days=30)

                for _uuid, _schedule in self._pl_cron_schedules.copy().items():
                    if _schedule['next'] and self._pl_cron_check_time >= _schedule['next']:
                        _puuid = None
                        try:
                            if not (_pl := Pipeline.query.filter_by(uuid=_uuid).first()):
                                logger.info('(Scheduler) Remove cron schedule for non-existing pipeline %s', _uuid)
                                self._pl_cron_schedules.pop(_uuid, None)
                                continue
                            logger.info('(Scheduler) Run cron schedule for pipeline %s', _uuid)
                            _schedule['next'] = self._get_pl_cron_next(_schedule['cron'], _uuid)
                            TaskOperator.project_pipeline_run(
                                _pl.owner, puuid=(_puuid := _pl.project.uuid), pluuid=_uuid,
                                comment='Triggered by system', use_track=False
                            )
                        except Exception as e:
                            auditlog.error(None, auditlog.PIPELINE, 'Auto start pipeline {0} failed: {1}',
                                           (_uuid, e), project=_puuid)

            time.sleep(5)

    def update_pipeline_cron_scheduler(self, partial_updates: Optional[dict] = None):
        """Update pipeline cron scheduler

        It does partial update when `partial_updates` is given; otherwise it fully updates
        according from all pipeline DB records.

        Args:
          partial_updates: {pl_uuid: cron_utc}. When cron_utc is empty, it removes that pipeline's schedling.
        """

        complete_updates = None
        if partial_updates is None:
            with commit_on_close():
                complete_updates = {
                    pl.uuid: {'cron': pl.crontab, 'next': self._get_pl_cron_next(pl.crontab, pl.uuid)}
                    for pl in Pipeline.query.filter_by(trigger_time=Pipeline.TRIGGER_CRON).all()
                    if pl.crontab
                }

        with self._pl_cron_lock:
            if partial_updates is None:
                logger.debug('(Scheduler) Full update pipeline cron scheduler')
                self._pl_cron_schedules = complete_updates
            else:
                logger.debug('(Scheduler) Partial update pipeline cron scheduler')
                for _uuid, _cron in partial_updates.items():
                    if _cron:
                        self._pl_cron_schedules[_uuid] = {'cron': _cron, 'next': self._get_pl_cron_next(_cron, _uuid)}
                    else:
                        self._pl_cron_schedules.pop(_uuid, None)

    def _get_pl_cron_next(self, cron_utc, pl_uuid) -> Optional[datetime]:
        try:
            return croniter(
                cron_utc, start_time=self._pl_cron_check_time,
                max_years_between_matches=1, hash_id=pl_uuid.encode('UTF-8'), ret_type=datetime
            ).get_next()
        except CroniterBadDateError:
            # cannot find the next date within the max-year window
            return None
