import itertools
import json
import math
import numpy as np
import re
import threading
import time
import typing
from collections import OrderedDict
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, Iterable, List, Optional, Tuple, Union

from mlsteam.core.const import K8S_PROJECT_LABEL, K8S_TASK_UUID_LABEL, K8S_TASK_VERSION_LABEL
from mlsteam.core.container import get_system_limit
from mlsteam.core.models.host import Host
from mlsteam.core.models.user import User
from mlsteam.core.plan import Plan, PlanCache, plan_list
from mlsteam.core.flavor import flavor_get_resources
from mlsteam.utils import parse_k8s_size
from mlsteam.sysconfig import config_get
from mlsteam.exception.resource import OutOfResource, ResourceAllocateError
from mlsteam.exception.base import MLSteamException
from mlsteam.core.utils import auditlog
from mlsteam.log import logger


GPU_NAME_CATEGORY = {}


def is_subset(parent, child):
    """Checker whether child is subset of parent"""
    parent = set(parent)
    child = set(child)
    return child.issubset(parent)


def check_tags(worker_tags, flavor_tags):
    # for allow-none-tag-flavor-go-any-host 2022/11/02
    # Deny case A:
    # 1. worker_tags == None
    # 2. flavor_tags != None
    if not worker_tags and flavor_tags:
        return False  # Deny!

    # Deny case B:
    # 1. worker_tags != None
    # 2. flavor_tags != None
    # 3. flavor_tags.is_subset(worker_tags) is False
    if worker_tags and flavor_tags:
        if not is_subset(worker_tags, flavor_tags):
            return False  # Deny!
    return True  # Pass!


def norm_gpu_type(raw_gtype: str) -> str:
    return 'any' if raw_gtype.lower() == 'any' else raw_gtype


class DefaultScoringStrategy:
    _max = None
    weight = dict(cpu=1, memory=1, gpu=0)
    failure_weight = 0.5  # penalty
    shape_points = [(0, 10), (30, 10), (80, 1), (100, 0)]  # util, score
    # 10 score
    # |* *
    # |    *
    # |      *
    # |        ** 100 util
    # |------------

    @classmethod
    def max(cls):
        if not cls._max:
            cls._max = max([i[1] for i in cls.shape_points])
        return cls._max

    @classmethod
    def _get_shape_value(cls, x):
        x_points, y_points = list(zip(*cls.shape_points))
        return np.interp(min(x, 100), np.array(x_points), np.array(y_points))

    @classmethod
    def _get_weighted_score(cls, weight, usage, capacity):
        if not weight:
            return 0
        if capacity <= 0:  # avoid divide by zero
            return 0
        return weight * cls._get_shape_value(math.floor(100 * usage / capacity))

    @classmethod
    def _get_weight_bias(cls, data):
        weight_min = 0
        for t in ['cpu', 'memory', 'gpu']:
            if t in cls.weight and t in data and data[t]['req']:
                weight_min = min(weight_min, cls.weight[t])
        if weight_min >= 0:
            return 0
        return abs(weight_min)  # if weight contains negtive

    @classmethod
    def _check_capacity(cls, data):
        for t in ['cpu', 'memory', 'gpu']:
            if t in cls.weight and t in data and data[t]['req']:
                if data[t]['max'] <= 0:
                    return False
        return True

    @classmethod
    def compute(cls, data, failure=0):
        # check have any request
        if sum([i['req'] for i in data.values()]) <= 0:
            return cls.max()  # every node should be max score
        if not cls._check_capacity(data):
            return 0  # impossible allocate to this node
        weighted_score = 0
        weight_sum = 0
        weight_bias = cls._get_weight_bias(data)
        for t in ['cpu', 'memory', 'gpu']:
            if t in cls.weight and t in data and data[t]['req']:
                _weight = cls.weight[t] + weight_bias
                weight_sum += _weight
                weighted_score += cls._get_weighted_score(
                    _weight, data[t]['now'] + data[t]['req'], data[t]['max'])
        if weight_sum == 0:  # avoid divide by zero
            return cls.max()  # every node should be max score
        weighted_score = min(cls.max(), weighted_score / weight_sum)
        if cls.failure_weight:
            weighted_score -= failure * cls.failure_weight
        return weighted_score

    @classmethod
    def scoring(cls, *args, **kwargs):
        score = cls.compute(*args, **kwargs)
        return math.floor(score * 100) / 100.0


class LeastAllocatedStrategy(DefaultScoringStrategy):
    shape_points = [(0, 10), (100, 0)]  # util, score


class MostAllocatedStrategy(DefaultScoringStrategy):
    shape_points = [(0, 0), (100, 10)]


class Resource(object):
    """
    Stores information about which tasks are using a resource
    """

    class ResourceAllocation(object):
        """
        Marks that a task is using [part of] a resource
        """

        def __init__(self, uid, value):
            """
            Arguments:
            task -- which task is using the resource
            value -- how much of the resource is being used
            """
            self.uid = uid
            self.value = value

    def __init__(self, identifier=None, name=None, max_value=1, ratio=1, **kwargs):
        """
        Keyword arguments:
        identifier -- some way to identify this resource
        max_value -- a numeric representation of the capacity of this resource
        kwargs -- id, name, memory (for gpu)
        """
        if identifier is None:
            self.identifier = id(self)
        else:
            self.identifier = identifier
        if name is None:
            self.name = self.identifier
        else:
            self.name = name
        self.max_value = max_value
        self.allocations = []
        self.ratio = ratio

        for key, value in kwargs.items():
            try:
                value = float(value)
            except Exception:  # nosec
                pass
            setattr(self, key, value)

    @property
    def capacity(self):
        return float(self.max_value * self.ratio)

    def remaining(self, exclude_allocated_tasks=None):
        """
        Returns the amount of this resource that is not being used
        """
        used = 0
        if exclude_allocated_tasks is None:
            exclude_allocated_tasks = []
        elif isinstance(exclude_allocated_tasks, str):
            exclude_allocated_tasks = [exclude_allocated_tasks]

        used = sum((a.value for a in self.allocations
                    if a.uid not in exclude_allocated_tasks))
        return max(self.capacity - used, 0)

    def used(self, exclude_allocated_tasks=None):
        return self.capacity - self.remaining(exclude_allocated_tasks)

    def allocate(self, uid, value, force=False):
        """
        A uuid is requesting to use this resource

        force: skip resource capacity checking (useful for external schedulers)
        """
        if not force and self.remaining() - value < 0:
            raise OutOfResource(f"Resource is already maxed out at {self.remaining}/{self.capacity}")

        self.allocations.append(self.ResourceAllocation(uid, value))
        # For GPU case
        if hasattr(self, 'index'):
            return self.identifier, self.index
        return self.identifier

    def deallocate(self, uid, use_prefix=False):
        """
        The task has finished using this resource
        """
        changed = False
        for i in self.allocations.copy():
            if uid == i.uid or (use_prefix and i.uid.startswith(uid)):
                self.allocations.remove(i)
                changed = True
        return changed

    def holders(self, exclude=None):
        total_holders = []
        for a in self.allocations.copy():
            if exclude and a.uid == exclude:
                continue
            total_holders.append(a.uid)
        return total_holders

    def holder_allocated(self, uid):
        for a in self.allocations.copy():
            if a.uid == uid:
                return a.value
        return 0


class GpuResource(object):
    class GPU(object):
        def __init__(self, uid, index, name, memory, mig, device_uuid,
                     instance_id, disable, slice_instance, capacity, mig_profile):
            self.uuid = uid
            self.index = index
            self.name = name
            self.memory = memory
            self.mig = mig  # whether supports MIG
            # list of profile id, ex. [9, 9] or []...
            self.mig_profile = mig_profile
            self.device_uuid = device_uuid  # (MIG) Device uuid
            self.instance_id = instance_id  # (MIG) GPU instance id
            self.disable = disable
            self.allocated_uid = None
            self.slice_instance = slice_instance
            self._capacity = capacity

        def mig_profile_to_policy(self):
            gb = int(self.memory) // (1024 * 8)
            profile_2_policy = {
                "all-disabled": 0,
                "all-3g.71gb": 1,
                "all-2g.35gb": 2,
                "all-balanced": 3,
                "all-1g.18gb": 6
            }

            if not self.mig_profile:
                return 0

            return profile_2_policy[self.mig_profile]

        def serialize(self):
            return {
                'index': self.index, 'uuid': self.uuid, 'name': self.name,
                'memory': self.memory, 'mig': self.mig, 'disable': self.disable,
                'mig_policy': self.mig_profile_to_policy(), 'allocated_uid': self.allocated_uid,
                'capacity': self.capacity, 'slice_instance': self.slice_instance,
            }

        def spec(self):
            return {
                'index': self.index,
                'name': self.name,
                'memory': self.memory,
                'mig': self.mig,
            }

        @property
        def capacity(self):
            if self.disable:
                return 0
            return self._capacity

    def __init__(self):
        # {'GPU-87da3a40-50b8-04da-80b4-ca69a99d14b0': GPU<object>}
        self._gpus: typing.OrderedDict[str, GpuResource.GPU] = OrderedDict()

    def name_to_profile(self, name):
        # name "NVIDIA A100-PCIE-40GB MIG 3G.20GB"
        # NVIDIA H100 80GB HBM3 MIG 1G.10GB+me
        # vcore, ram fraction, +me
        mapping = {  # H100, A100
            (7, 1, 0): 0,
            (4, 2, 0): 5,
            (3, 2, 0): 9,
            (2, 4, 0): 14,
            (1, 4, 0): 15,
            (1, 8, 0): 19,
            (1, 8, 1): 20,
        }
        if 'A30' in name:
            mapping = {
                (4, 1, 0): 0,
                (2, 2, 0): 5,
                (2, 2, 1): 6,
                (1, 4, 0): 14,
                (1, 4, 1): 21,
            }
        ret = re.search(r'([0-9]+)gb.* mig ([0-9])g\.([0-9]+)gb(\+me)?', name, re.I)
        if ret:
            ram, vcore, vram, me = ret.groups()
            ram, vcore, vram, me = int(ram), int(vcore), int(vram), int(bool(me))
            ram_frac = int(ram / vram)
            profile_id = mapping.get((vcore, ram_frac, me))
            if profile_id:
                return profile_id
        auditlog.error('SYSTEM', auditlog.SYSTEM, "Get an unsupport MIG device name {0}, please check it", (name))
        return -1

    def add(self, uid, index, name, memory, mig=False, device_uuid=None, instance_id=None, disable=False,
            slice_instance=False, capacity=1, mig_profile=''):
        self._gpus[uid] = self.GPU(uid, index, name, memory, mig, device_uuid, instance_id, disable, slice_instance,
                                   capacity, mig_profile)

    def serialize(self):
        return [gpu.serialize() for _, gpu in self._gpus.items()]

    def spec(self):
        return {gpu.uuid: gpu.spec() for gpu in self._gpus.values()}

    def get(self, _uid) -> typing.Optional[GPU]:
        return self._gpus.get(_uid)

    def get_all(self) -> List[Tuple[str, GPU]]:
        return [(guid, gpu) for (guid, gpu) in self._gpus.items()]

    def get_available(self, exclude_allocated_tasks=None,
                      override_disable: Optional[Dict[str, bool]] = None,
                      override_mig: Optional[Dict[str, int]] = None) -> List[Tuple[str, GPU]]:
        available = []
        if exclude_allocated_tasks is None:
            exclude_allocated_tasks = []
        elif isinstance(exclude_allocated_tasks, str):
            exclude_allocated_tasks = [exclude_allocated_tasks]
        override_disable = (override_disable or {}).copy()  # we may rewrite it later (for MIG)
        override_mig = override_mig or {}

        for guid, mig_policy in override_mig.items():
            if (p_index := next((_gpu.index for _gpu in self._gpus.values() if _gpu.uuid == guid), None)) is None:
                continue
            disable_parent, disable_instance = (
                (False, True) if mig_policy == 0  # disable MIG
                else (True, False)  # enable/change MIG
            )
            override_disable[guid] = disable_parent
            for _guid, _gpu in self._gpus.items():
                if _gpu.index.startswith(p_index + ':'):  # beloning instances (if exists)
                    override_disable[_guid] = disable_instance

        for guid, gpu in self._gpus.items():
            if override_disable.get(guid, gpu.disable):
                continue
            if gpu.allocated_uid and gpu.allocated_uid not in exclude_allocated_tasks:
                continue
            available.append((guid, gpu))
        # logger.debug("gpu get_available: {}".format([(a[0], a[1].serialize()) for a in available]))
        return available

    def update(self, guid, allocated_uid=None, disable=None):
        for guuid, gpu in self._gpus.items():
            if guid == guuid:
                if allocated_uid is not None:
                    gpu.allocated_uid = allocated_uid
                if disable is not None:
                    gpu.disable = disable
                break

    # def is_inuse(self, guid):
    #     mig_index = None
    #     for guuid, gpu in self._gpus.items():
    #         if guid == guuid:
    #             if gpu.allocated_uid:
    #                 return True
    #             # check for MIG partitioned GPUs
    #             if gpu.mig and gpu.mig_profile:
    #                 mig_index = gpu.index
    #     if mig_index:
    #         for _, gpu in self._gpus.items():
    #             if gpu.index.startswith(mig_index+":") and gpu.allocated_uid:
    #                 return True
    #     return False

    def any_inuse(self):
        for gpu in self._gpus.values():
            if gpu.allocated_uid:
                return True
        return False

    @classmethod
    def _update_gpu(cls, data: dict, gpu_type, val=1):
        data[gpu_type] = data.get(gpu_type, 0) + val
        data['any'] = data.get('any', 0) + val

    @property
    def capacity(self) -> Dict[str, int]:
        # total gpu num exclude disabled
        ret = {'any': 0}
        for _, gpu in self._gpus.items():
            self._update_gpu(ret, gpu.name, gpu.capacity)
        return ret

    def remaining(self, exclude_allocated_tasks=None,
                  override_disable: Optional[Dict[str, bool]] = None,
                  override_mig: Optional[Dict[str, int]] = None) -> Dict[str, int]:
        ret = {'any': 0}
        for _, gpu in self.get_available(exclude_allocated_tasks, override_disable, override_mig):
            self._update_gpu(ret, gpu.name, gpu.capacity)
        return ret

    def used(self, exclude_allocated_tasks=None) -> Dict[str, int]:
        _remaining = self.remaining(exclude_allocated_tasks)
        ret = {gpu_name: gpu_capacity - _remaining.get(gpu_name, 0)
               for gpu_name, gpu_capacity in self.capacity.items()}
        return ret

    def remaining_type(self, gpu_type, exclude_allocated_tasks=None):
        _sum = 0
        for _, gpu in self.get_available(exclude_allocated_tasks):
            if gpu.name == gpu_type:
                _sum += 1
        return _sum

    def allocate(self, uid, gpu_num, gpu_type, force=False):
        if not gpu_num or not gpu_type:
            return []
        if gpu_type.lower() == "any":
            gpu_type = "any"
        if not force and (_remaining := self.remaining()[gpu_type]) - gpu_num < 0:
            raise OutOfResource(f"Resource is already maxed out at {_remaining}/{self.capacity[gpu_type]}")
        allocated_gpu = []
        for _, gpu in self.get_available():
            if gpu.slice_instance and gpu_type == "any":
                continue
            if gpu_type != "any" and gpu.name != gpu_type:
                # skip non-match gpu type
                continue
            gpu.allocated_uid = uid
            allocated_gpu.append((gpu.uuid, gpu.index))
            if len(allocated_gpu) == gpu_num:
                break
        if not force and len(allocated_gpu) != gpu_num:
            for guid, _ in allocated_gpu:
                self.get(guid).allocated_uid = None
            raise OutOfResource("GPU resource not enough")
        return allocated_gpu

    def deallocate(self, uid, use_prefix=False):
        changed = False
        for (_, gpu) in self.get_all():
            if (gpu.allocated_uid == uid or
                    (use_prefix and gpu.allocated_uid and gpu.allocated_uid.startswith(uid))):
                # remove allocated identifier
                gpu.allocated_uid = None
                changed = True
        return changed

    def holders(self, exclude=None):
        total_holders = []
        for guid, gpu in self.get_all():
            if exclude and guid == exclude:
                continue
            if gpu.allocated_uid and gpu.allocated_uid not in total_holders:
                total_holders.append(gpu.allocated_uid)
        return total_holders

    def holder_allocated(self, uid):
        hold_gpus = []
        for _, gpu in self.get_all():
            if gpu.allocated_uid == uid:
                hold_gpus.append(gpu.serialize())
        return hold_gpus


class Worker(object):

    def __init__(self, ip, hostname, cpu_resource: Resource = None, memory_resource: Resource = None,
                 gpu_resource: GpuResource = None, cluster_uuid='', **kwargs):
        self.sid = str()
        self.ip = ip
        self.hostname = hostname
        self.cluster_uuid = cluster_uuid
        self.lock = threading.RLock()

        # host resource
        self.worker_gpus = gpu_resource
        self.cpu = cpu_resource
        self.memory = memory_resource
        self.tags = {}  # host.tags
        self.roles = []  # worker, edge
        self.videos = []  # /dev/video0, /dev/video1
        self.specs = []  # NVIDIA Jetson Nano Developer Kit
        self.docker_fstype = ''
        self.platform = {}

        # host heartbeat (contains tasks)
        self.tasks_need_resume = True  # if host join, resume tasks base on self.tasks_heartbeat
        self.tasks_heartbeat = {}
        self._last_heartbeat = datetime.min.replace(tzinfo=timezone.utc)
        self.get_heartbeat_now = True

        # host status
        self._authorized = False
        self._connected = False  # agent is connected
        self._is_error = False  # if node_setup fail
        self.components = {}
        self._is_master = False
        self.task_failure_count = 0
        self.netdata_temperature_url = 'api/v1/data?chart=sensors.coretemp-isa-0000_temperature&after=-1'

    @property
    def ready(self):
        # Basic
        if not self.alive:
            return False
        if not self.container_connected:
            return False
        if 'edge' in self.roles:
            return True
        if not self.term_service_connected:
            return False
        # if not self.netdata_connected:
        #     return False
        return True

    @property
    def alive(self):
        return (self._connected and self._authorized)

    @property
    def netdata_connected(self):
        return self.components.get('netdata') == 'running'

    @property
    def term_service_connected(self):
        return self.components.get('term_service') == 'running'

    @property
    def container_connected(self):
        # reachable, master or in cluster
        if self._is_master:
            return True
        return self.components.get('swarm') == 'running'

    def change_ratio(self, rtype, ratio):
        try:
            if rtype in ["cpu", "memory"]:
                resource = getattr(self, rtype)
                resource.ratio = ratio
            elif rtype == "gpu":
                for _, gpu in self.worker_gpus.items():
                    gpu.ratio = ratio
        except Exception as e:
            raise MLSteamException('change worker {} {} ratio failed'.format(self.hostname, rtype)) from e

    def resource_spec(self, rtype):
        if rtype == "gpu":
            return self.worker_gpus.serialize()
        return []

    def resource_score(self, requested, exclude_uuid=None, scoring_strategy=DefaultScoringStrategy):
        data = {}
        data['cpu'] = dict(
            now=self.cpu.used([exclude_uuid]),
            req=requested.get('cpu', 0),
            max=self.cpu.capacity)
        data['memory'] = dict(
            now=self.memory.used([exclude_uuid]),
            req=requested.get('memory', 0),
            max=self.memory.capacity)
        data['gpu'] = dict(
            now=self.worker_gpus.used([exclude_uuid])['any'],
            req=requested.get('gpu', 0),
            max=self.worker_gpus.capacity['any'])
        return scoring_strategy.scoring(data, self.task_failure_count)

    def spec(self):
        return {
            'cpu': self.cpu.capacity,
            'memory': self.memory.capacity,
            'gpus': self.worker_gpus.spec(),
        }

    def resource_list(self, reload=False, exclude_allocated_tasks=None,
                      override_gpu_disable: Optional[Dict[str, bool]] = None,
                      override_gpu_mig: Optional[Dict[str, int]] = None):
        resources = {}
        if self.cpu:
            resources['cpu'] = {
                'remaining': self.cpu.remaining(exclude_allocated_tasks),
                'total': self.cpu.capacity,
                'unit': 'core',
            }
        if self.memory:
            resources['memory'] = {
                'remaining': self.memory.remaining(exclude_allocated_tasks),
                'total': self.memory.capacity,
                'unit': 'mb',
            }
        if self.worker_gpus:
            gpu_spec = []
            for _, gpu in self.worker_gpus.get_all():
                if gpu.allocated_uid:
                    continue
                gpu_spec.append(gpu.serialize())
            resources['gpu'] = {
                'remaining': self.worker_gpus.remaining(exclude_allocated_tasks,
                                                        override_gpu_disable, override_gpu_mig),
                'total': self.worker_gpus.capacity,
                'spec': gpu_spec  # details of each GPU
            }
        else:
            resources['gpu'] = {'remaining': 0, 'total': 0, 'spec': []}
        resources['holders'] = []
        for holder in self.cpu.holders(exclude_allocated_tasks):
            resources['holders'].append({
                'uuid': holder,
                'cpu': self.cpu.holder_allocated(holder),
                'memory': self.memory.holder_allocated(holder),
                'gpu': self.worker_gpus.holder_allocated(holder)
            })
        return resources

    def available_resource_list(self, exclude_allocated_tasks=None):
        resources = {}
        if self.cpu:
            resources['cpu'] = {
                'remaining': self.cpu.remaining(exclude_allocated_tasks),
                'total': self.cpu.capacity,
                'unit': 'core',
            }
        if self.memory:
            resources['memory'] = {
                'remaining': self.memory.remaining(exclude_allocated_tasks),
                'total': self.memory.capacity,
                'unit': 'mb',
            }
        if self.worker_gpus:
            resources['gpu'] = {
                'remaining': self.worker_gpus.remaining(exclude_allocated_tasks),
                'total': self.worker_gpus.capacity,
                'spec': [gpu.serialize() for _, gpu in self.worker_gpus.get_available(exclude_allocated_tasks)]
            }
        else:
            resources['gpu'] = {'remaining': {'any': 0}, 'total': 0, 'spec': []}
        resources['holders'] = []
        for holder in self.cpu.holders(exclude_allocated_tasks):
            resources['holders'].append({
                'uuid': holder,
                'cpu': self.cpu.holder_allocated(holder),
                'memory': self.memory.holder_allocated(holder),
                'gpu': self.worker_gpus.holder_allocated(holder)
            })
        return resources

    def check(self, uid, cpu_core=0, mem_mb=0, gpu_num=0, gpu_type="any", exclude_allocated_tasks=[]):
        if (remaining := self.cpu.remaining(exclude_allocated_tasks)) < cpu_core:
            raise ResourceAllocateError(
                f"[{self.hostname}] Out of Cpu resources, remain: {remaining}, need: {cpu_core}",
                task=uid, device="Cpu")
        if (remaining := self.memory.remaining(exclude_allocated_tasks)) < mem_mb:
            raise ResourceAllocateError(
                f"[{self.hostname}] Out of Memory resources, remain: {remaining}, need: {mem_mb}",
                task=uid, device="Memory")
        if gpu_type:
            if gpu_type.lower() == "any":
                gpu_type = "any"
            if remaining := self.worker_gpus.remaining(exclude_allocated_tasks)[gpu_type] < gpu_num:
                raise ResourceAllocateError(
                    f"[{self.hostname}] Out of GPU resources, remain: {remaining}, need: {gpu_num}",
                    task=uid, device="GPU")

    def allocate(self, uid, cpu_core=.0, mem_mb=.0, gpu_num=0, gpu_type="any", force=False) -> dict:
        cpu_core = float(cpu_core)
        mem_mb = float(mem_mb)
        gpu_num = int(gpu_num)
        cpu_set = []  # Manta NOTE: no longer determine cpu sets
        with self.lock:
            if not force:
                self.check(uid, cpu_core, mem_mb, gpu_num, gpu_type)
            # allocate resources
            if cpu_core:
                self.cpu.allocate(uid, cpu_core, force=force)
            if mem_mb:
                self.memory.allocate(uid, mem_mb, force=force)
            gpu_allocated = []
            if gpu_num:
                gpu_allocated = self.worker_gpus.allocate(uid, gpu_num, gpu_type, force=force)

        requested_resources = {
            'hostname': self.hostname,
            'gpus': gpu_allocated,
            'cpus': {'num': cpu_core, 'set': ','.join(map(str, cpu_set))},
            'memory': mem_mb
        }
        return requested_resources

    def reallocate(self, uid, resources: dict):
        '''
        {
            'hostname': 'quantumcloud4',
            'gpus': [
                ('GPU-ae66400d-8ca1-41ab-26fd-1226658040b6', 2),
                ('GPU-b0d21d4c-41d9-cff4-43f8-8351cee9d6c8', 3),
                ('GPU-659b1d57-fcc5-ba40-4c8e-2a78a33afb88', 4),
                ('GPU-4a250d52-9f77-3c3e-9b2e-0dd549ed027c', 5)
            ],
            'cpus': {
                'num': 32,
                'set': '60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,0,1,2,3,4,5,6,7,8,9,10,11'
            },
            'memory': 344064
        }
        '''
        with self.lock:
            # allocate resources
            if 'cpus' in resources:
                cpu_core = resources['cpus'].get('num')
                if cpu_core:
                    self.cpu.allocate(uid, cpu_core)
            if 'memory' in resources:
                mem_mb = resources['memory']
                if mem_mb:
                    self.memory.allocate(uid, mem_mb)
            if 'gpus' in resources:
                gpu_list = resources.get('gpus')
                if gpu_list:
                    gpu_uuid_list = [g[0] for g in gpu_list]
                    for guid, gpu in self.worker_gpus.get_all():
                        if guid in gpu_uuid_list:
                            gpu.allocated_uid = uid

    def find_resource(self, requested, exclude_allocated_tasks=None) -> dict:
        res_host = self.available_resource_list(exclude_allocated_tasks)
        logger.debug("find host resources(exclude: %s): %s", exclude_allocated_tasks, res_host)
        # GPU is first checked resource, refer to flavor_get_resources()
        for res_type in ["gpu", "cpu", "memory"]:
            if res_type == 'gpu':
                req = requested.get('gpu', 0)
                gtype = requested.get('gpu_type', 'any')
                res = res_host.get('gpu', {}).get('remaining', {}).get(norm_gpu_type(gtype), 0)
            else:
                req = requested.get(res_type, 0)
                res = res_host.get(res_type, {}).get('remaining', 0)
            if res < req:
                raise OutOfResource(
                    f"[{self.hostname}] Out of f{res_type.upper()} resources", device=res_type.upper())
        # check GPU type
        logger.debug("iterate find gpu_type for {} on {}".format(
                     requested.get("gpu_type"), res_host.get("gpu", {}).get("spec", [])))
        if requested.get("gpu_type", "any").lower() != "any":
            # count for gpu_type
            gtype = norm_gpu_type(requested.get("gpu_type"))
            # host gpu type less than requested
            if res_host['gpu']['remaining'][gtype] < requested.get("gpu", 0):
                raise OutOfResource(
                    f"[{self.hostname}] Out of GPU resources", device="GPU")

        require_video_enable = requested.get('video_enable') or False
        resources_found = requested.copy()
        if 'cpu' in resources_found:
            resources_found['cpus'] = {'num': resources_found.pop('cpu')}
        if 'gpu' in resources_found:
            resources_found.pop('gpu')
            if (req := requested.get('gpu', 0)):
                gtype = norm_gpu_type(requested.get('gpu_type', 'any'))
                specs = [s for s in res_host['gpu']['spec'] if gtype == 'any' or gtype == s['name']]
                resources_found['gpus'] = [
                    {k: v for k, v in s.items() if k not in ('index', 'uuid')}  # remove unused keys
                    for s in specs[:req]
                ]
            else:
                resources_found['gpus'] = []
        # Add video device for edge case
        if require_video_enable and self.videos:
            resources_found['videos'] = self.videos.copy()
        return resources_found

    def update_gpu_resource(self, new_gpu_resource: GpuResource):
        if new_gpu_resource:
            for (guid, _) in new_gpu_resource.get_all():
                gpu = self.worker_gpus.get(guid)
                if gpu:
                    new_gpu_resource.update(guid, allocated_uid=gpu.allocated_uid)
            self.worker_gpus = new_gpu_resource

    def deallocate(self, uid, use_prefix=False):
        with self.lock:
            self.cpu.deallocate(uid, use_prefix=use_prefix)
            self.memory.deallocate(uid, use_prefix=use_prefix)
            self.worker_gpus.deallocate(uid, use_prefix=use_prefix)

    def check_tags(self, flavor_tags):
        return check_tags(worker_tags=self.tags, flavor_tags=flavor_tags)


class KubernetesWorker(Worker):

    POD_DEL_PRESERVATION_TIME = 30 * 60  # time to preserve deleted pod info

    class PodPhase(Enum):
        PENDING = 'Pending'
        RUNNING = 'Running'
        SUCCEEDED = 'Succeeded'
        FAILED = 'Failed'
        UNKNOWN = 'Unknown'

        @classmethod
        def from_str(cls, expr: str):
            return cls(expr.capitalize())

        @property
        def occupy_resources(self):
            return self in (self.__class__.PENDING, self.__class__.RUNNING)

        @property
        def occupy_no_resources(self):
            return self in (self.__class__.SUCCEEDED, self.__class__.FAILED)

    @dataclass
    class PodInfo:
        hostname: str
        namespace: str
        name: str
        labels: dict[str, str]
        annotations: dict[str, str]
        phase: 'KubernetesWorker.PodPhase'
        message: str
        reason: str
        conditions: list[dict]
        container_specs: list[dict]
        container_statuses: list[dict]
        deleted: bool = False

        @classmethod
        def from_dict(cls, raw_dict: dict):
            return cls(
                hostname=raw_dict['spec'].get('nodeName', ''),
                namespace=raw_dict['metadata']['namespace'],
                name=raw_dict['metadata']['name'],
                labels=raw_dict['metadata'].get('labels', {}),
                annotations=raw_dict['metadata'].get('annotations', {}),
                phase=KubernetesWorker.PodPhase.from_str(raw_dict['status']['phase']),
                message=raw_dict['status'].get('message', ''),
                reason=raw_dict['status'].get('reason', ''),
                conditions=raw_dict['status'].get('conditions', []),
                container_specs=raw_dict['spec'].get('containers', []),
                container_statuses=raw_dict['status'].get('containerStatuses', [])
            )

        @property
        def proj_uuid(self) -> str:
            return self.labels.get(K8S_PROJECT_LABEL, '')

        @property
        def task_uuid(self) -> str:
            return self.labels.get(K8S_TASK_UUID_LABEL, '')

        @property
        def task_version(self) -> str:
            return self.labels.get(K8S_TASK_VERSION_LABEL, '')

        @property
        def pod_ready(self) -> bool:
            try:
                matched_cnd = [c.get('status', '').lower() for c in self.conditions if c.get('type') == 'Ready'][0]
                return matched_cnd == 'true'
            except Exception:
                return False

        def iter_container_resources(self):
            for cs in self.container_specs:
                cs_rsc = cs.get('resources', {})
                yield cs.get('name'), {'limits': cs_rsc.get('limits', {}), 'requests': cs_rsc.get('requests', {})}

        def to_deleted(self):
            self.deleted = True
            return self

        def __eq__(self, other):
            return self.namespace == other.namespace and self.name == other.name

        def __str__(self):
            return '<Pod namespace={}, name={}>'.format(self.namespace, self.name)

    class PodResources:
        """Calculate resource consumption for a pod

        Calculation methods:
        1. Pod is in an active phase, and all containers have resource limits (most reliable):
           `sum(limits per container)`
        2. Condition 1. unsatisfied, with task resources:
           `max(sum(limits or requests per container), task resources)`
        3. Condition 1. unsatisfied, without task resources:
           `max(sum(limits or requests per container), default value)`

        NOTE:
        1. The calculation assumes task resource constraints are applied to a single pod.
        Calculation should be enhanced when resource constraints are applied to a set of pods
        (possibly with namespace resource limits).
        2. It only supports NVIDIA GPU Operator now.
        """
        RES_TYPES = ('cpu_core', 'mem_mb', 'gpu_num')
        K8S_RES_TYPES = {'cpu_core': 'cpu', 'mem_mb': 'memory'}

        def __init__(self, worker: 'KubernetesWorker', pod: 'KubernetesWorker.PodInfo'):
            self.worker = worker
            self.pod = pod
            self.res = {rt: 0 for rt in self.RES_TYPES} | {'gpu_type': 'any'}
            self.missing_limits = set()
            self._apply_container_resources()
            self._apply_task_or_default_resources()

        def _apply_container_resources(self):
            if not self.pod.phase.occupy_resources:
                self.missing_limits = set(self.RES_TYPES)
                return
            for _, c_res in self.pod.iter_container_resources():
                for rt in self.RES_TYPES:
                    if (res_val := self._get_k8s_resource_spec(c_res['limits'], rt)) is not None:
                        self.res[rt] += res_val
                    else:
                        self.missing_limits.add(rt)
                        if (res_val := self._get_k8s_resource_spec(c_res['requests'], rt)) is not None:
                            self.res[rt] += res_val
            if self.res['gpu_num'] > 0 and (worker_gpus := self.worker.worker_gpus.get_all()):
                self.res['gpu_type'] = worker_gpus[0][1].name

        def _apply_task_or_default_resources(self):
            from mlsteam.core.tasks.helper import TaskCache
            if self.missing_limits:
                if not self.pod.task_uuid or not (task := TaskCache.get(self.pod.task_uuid)):
                    res = {
                        'cpu_core': 0.05,
                        'mem_mb': 10,
                        'gpu_num': 0
                    }
                else:
                    res = {
                        'cpu_core': task.resources.get('cpus', {}).get('num', 1),
                        'mem_mb': task.resources.get('memory', 102400) / 1_000_000,
                        'gpu_num': len(task.resources.get('gpus', []))
                    }
                for rt in self.missing_limits:
                    self.res[rt] = max(self.res[rt], res[rt])

        @classmethod
        def _get_k8s_resource_spec(cls, c_rspec: dict, rt: str):
            match rt:
                case 'cpu_core':
                    k8s_res = c_rspec.get('cpu')
                case 'mem_mb':
                    k8s_res = c_rspec.get('memory')
                case 'gpu_num':
                    # NOTE: support ROCm GPU in the future
                    k8s_res = sum(map(
                        int,
                        dict(filter(
                            lambda i: i[0].startswith("nvidia.com/"),
                            c_rspec.items()
                        )).values()
                    ))
            if k8s_res:
                return parse_k8s_size(k8s_res) / 1048576 if rt == 'mem_mb' else parse_k8s_size(k8s_res)
            return None

    def __init__(self, ip, hostname, cpu_resource: Resource = None,
                 memory_resource: Resource = None, gpu_resource: GpuResource = None,
                 cluster_uuid: str = '', k8s_allocatable: dict | None = None, k8s_conditions: list | None = None,
                 **kwargs):
        k8s_allocatable = k8s_allocatable or {}
        k8s_conditions = k8s_conditions or []
        super().__init__(
            ip=ip, hostname=hostname,
            cpu_resource=cpu_resource, memory_resource=memory_resource, gpu_resource=gpu_resource,
            cluster_uuid=cluster_uuid, k8s_allocatable=k8s_allocatable, k8s_conditions=k8s_conditions, **kwargs
        )
        self.k8s_allocatable = k8s_allocatable
        self.k8s_conditions = k8s_conditions
        self.k8s_pods: list[self.PodInfo] = []
        self.k8s_del_pods: list[tuple[self.PodInfo, float]] = []  # (pod info, del preservation timestamp)
        self.last_k8s_del_time = time.time()
        self.k8s_pods_lock = threading.RLock()

    @property
    def ready(self):
        return next((c['status'].lower() == 'true' for c in self.k8s_conditions if c['type'] == 'Ready'), False)

    @property
    def alive(self):
        return self.ready

    @property
    def netdata_connected(self):
        return False

    @property
    def term_service_connected(self):
        return False

    @property
    def container_connected(self):
        return False

    def deallocate_all(self):
        self.cpu.allocations.clear()
        self.memory.allocations.clear()
        for _, gpu in self.worker_gpus.get_all():
            gpu.allocated_uid = None

    def renew_pods(self, pods: list[PodInfo]):
        with self.k8s_pods_lock:
            orig_k8s_pods = self.k8s_pods
            self.k8s_pods = pods
            self._update_allocations(renewed=pods)
            deleted_pod_names = set((p.name for p in orig_k8s_pods)) - set((p.name for p in pods))
            deleted_pods = [p.to_deleted() for p in orig_k8s_pods if p.name in deleted_pod_names]
            unready_pods = [p for p in orig_k8s_pods if p.name not in deleted_pod_names and not p.pod_ready]
            del_preservation_time = time.time() + self.POD_DEL_PRESERVATION_TIME
            self.k8s_del_pods = [(p, del_preservation_time) for p in deleted_pods]
            return deleted_pods, unready_pods

    def add_pod(self, pod: PodInfo):
        with self.k8s_pods_lock:
            self.k8s_pods.append(pod)
            self._update_allocations(added=pod)

    def del_pod(self, pod: PodInfo):
        with self.k8s_pods_lock:
            with suppress(ValueError):
                self.k8s_pods.remove(pod)
                self.k8s_del_pods.append((pod.to_deleted(), time.time() + self.POD_DEL_PRESERVATION_TIME))
                self._update_allocations(deleted=pod)

    def update_pod(self, pod: PodInfo):
        with self.k8s_pods_lock:
            # clean up long deleted pods
            if (curr_time := time.time()) > self.last_k8s_del_time + 300:
                self.last_k8s_del_time = curr_time
                self.k8s_del_pods = [del_info for del_info in self.k8s_del_pods if curr_time > del_info[1]]
            try:
                index = self.k8s_pods.index(pod)
                self.k8s_pods[index] = pod
            except ValueError:
                self.k8s_pods.append(pod)
                self._update_allocations(added=pod)

    def list_pods(self, project_uuid=None, task_uuid=None, task_version=None):
        filters = {}
        if project_uuid:
            filters[K8S_PROJECT_LABEL] = project_uuid
        if task_uuid:
            filters[K8S_TASK_UUID_LABEL] = task_uuid
        if task_version:
            filters[K8S_TASK_VERSION_LABEL] = task_version
        with self.k8s_pods_lock:
            pod_iter = itertools.chain(self.k8s_pods, (x[0] for x in self.k8s_del_pods))
            if filters:
                return [p for p in pod_iter if
                        all((p.labels.get(label_name) == label_val for label_name, label_val in filters.items()))]
            return list(pod_iter)

    def get_uid_prefix(self, task_uuid=None, task_version=None, pod_namespace=None, pod_name=None) -> str:
        """Get complete or prefix of resource uid

        Resource uid format:
        - Resources consumed by a known task: `task:task_uuid[:task_version[:pod_namespace:pod_name]]`
        - Other resource comsuption: `other:pod_name`
        """
        if task_uuid:
            prefix = f'task:{task_uuid}:'
            if task_version:
                prefix += f'{task_version}:'
                if pod_namespace:
                    prefix += f'{pod_namespace}:{pod_name}'
        else:
            prefix = f'other:{pod_name}'
        return prefix

    def _update_allocations(self, renewed: list[PodInfo] | None = None,
                            added: PodInfo | None = None,
                            deleted: PodInfo | None = None,
                            updated: PodInfo | None = None):
        """Recalculate resource allocations by pod changes"""
        with self.k8s_pods_lock:
            if renewed:
                self.deallocate_all()
                for pod in renewed:
                    uid = self.get_uid_prefix(pod.task_uuid, pod.task_version, pod.namespace, pod.name)
                    self.allocate(uid=uid, **self.PodResources(self, pod).res, force=True)
            if added:
                uid = self.get_uid_prefix(added.task_uuid, added.task_version, added.namespace, added.name)
                self.allocate(uid=uid, **self.PodResources(self, added).res, force=True)
            if deleted:
                uid = self.get_uid_prefix(deleted.task_uuid, deleted.task_version, deleted.namespace, deleted.name)
                self.deallocate(uid=uid, use_prefix=True)
            if updated:
                try:
                    curr_index = self.k8s_pods.index(updated)
                    curr_pod = self.k8s_pods[curr_index]
                except ValueError:
                    logger.error('Failed to update resource allocations: no such pod %s', str(updated))
                    return
                if curr_pod.phase != updated.phase:
                    uid = self.get_uid_prefix(updated.task_uuid, updated.task_version, updated.namespace, updated.name)
                    self.deallocate(uid=uid, use_prefix=True)
                    self.allocate(uid=uid, **self.PodResources(self, updated).res, force=True)


class WorkerResources(object):

    def __init__(self):
        self.workers: typing.OrderedDict[str, Worker] = OrderedDict()
        self.ports = []

    def change_ratio(self, rtype, ratio):
        for _, worker in self.workers.items():
            try:
                worker.change_ratio(rtype, ratio)
            except Exception as e:
                logger.error(str(e))

    def get_worker(self, hostname: str):
        '''
        Get worker may not be ready
        '''
        return self.workers.get(hostname)

    def get_ready_worker(self, hostname: str):
        '''
        Get ready worker
        '''
        w = self.workers.get(hostname)
        if not w:
            return None
        if not w.ready:
            return None
        return w

    def list_ready_workers(self):
        return [(hostname, worker) for hostname, worker in self.workers.items() if worker.ready]

    def list_workers(self):
        return self.workers.copy().items()

    def get_scoring_strategy(self):
        schedule_strategy = config_get('schedule_strategy', '')
        if schedule_strategy.lower() == 'mostallocated':
            return 'MostAllocated'
        else:
            return 'LeastAllocated'

    def get_scoring_workers(self, requested, workers=None, exclude_uuid=None, scoring_strategy=None):
        if not workers:
            workers = self.list_ready_workers()

        # Scoring method from sysconfig
        if not scoring_strategy:
            scoring_strategy = self.get_scoring_strategy()
        scoring_strategy = MostAllocatedStrategy if scoring_strategy == 'MostAllocated' else LeastAllocatedStrategy

        workers_with_score = []
        for hostname, w in workers:
            workers_with_score.append({
                'score': w.resource_score(requested, exclude_uuid, scoring_strategy=scoring_strategy),
                'object': w,
                'hostname': hostname,
            })
        workers_with_score = sorted(workers_with_score, key=lambda x: x['score'], reverse=True)
        workers = list(map(lambda x: (x['hostname'], x['object']), workers_with_score))
        scores = list(map(lambda x: (x['hostname'], x['score']), workers_with_score))
        return workers, scores

    def sorted_scoring_workers(self, requested, exclude_uuid=None):
        workers = self.list_ready_workers()
        if len(workers) <= 1:
            return workers
        workers, scores = self.get_scoring_workers(requested, workers=workers, exclude_uuid=exclude_uuid)
        logger.debug('Worker Resource Score: {}'.format(scores))
        return workers

    def on_node_join(self, host: Host, **kwargs):
        from mlsteam.core.cluster import ClusterType
        global GPU_NAME_CATEGORY
        ip = host.ip
        hostname = host.hostname
        cluster_uuid = ''
        if host.in_cluster and host.cluster_hosts.cluster:
            cluster_uuid = host.cluster_hosts.cluster.uuid
        resources = host.resources
        authorized = host.authorize
        is_master = host.is_master
        tags = host.tags or {}
        roles = host.resources.get('roles', '')
        specs = host.resources.get('specs', '')
        docker_fstype = host.resources.get('docker_fstype', '')
        videos = host.resources.get('videos', [])
        platform = host.resources.get('platform', {})
        w = self.workers.get(hostname, None)

        # already joined
        if w and w._connected:
            if not (kwargs.get('joined_ok', False)):
                logger.warning("%s already in host cluster", hostname)
            return w
        # task_failure_count = w.task_failure_count if w else 0
        vcpu_ratio = float(get_system_limit('vcpu_ratio', 1))
        cpu = Resource(max_value=resources['cpu_cores'], ratio=vcpu_ratio)
        memory = Resource(max_value=resources['memory_mb'])
        logger.debug("host %s resources: vcpu->%s, mem->%s, gpu->%s",
                     hostname, cpu.capacity, memory.capacity, len(resources['gpus']))
        gpus = GpuResource()
        for gpu in resources['gpus']:
            gpus.add(
                gpu['uuid'],
                gpu['index'],
                gpu['name'],
                gpu['memory'],
                mig=gpu.get('mig', False),
                device_uuid=gpu.get('device_uuid'),
                instance_id=gpu.get('instance_id'),
                disable=gpu.get('disable', False),
                slice_instance=gpu.get('slice_instance', False),
                capacity=gpu.get('capacity', 1),
                mig_profile=gpu.get('mig_profile', 'all-disabled')
            )
            categories = gpu.get('categories', '')
            GPU_NAME_CATEGORY[gpu['name']] = categories
        if host.in_cluster and host.cluster_hosts.cluster.cluster_type == ClusterType.KUBERNETES:
            w = KubernetesWorker(ip, hostname, gpu_resource=gpus, cpu_resource=cpu, memory_resource=memory,
                                 cluster_uuid=cluster_uuid,
                                 k8s_allocatable=kwargs.get('k8s_allocatable', {}),
                                 k8s_conditions=kwargs.get('k8s_conditions', []))
        else:
            w = Worker(ip, hostname, gpu_resource=gpus, cpu_resource=cpu, memory_resource=memory)
        # reload disabled GPU from db
        blacklist = json.loads(host.resources_blacklist) if host.resources_blacklist else {}
        gpu_blist = blacklist.get('gpu', [])
        for guid in gpu_blist:
            w.worker_gpus.update(guid, disable=True)
        self.workers[hostname] = w
        if tags:
            w.tags = tags
        if roles:
            w.roles = roles
        if videos:
            w.videos = videos
        if platform:
            w.platform = platform
        if specs:
            w.specs = specs
        if docker_fstype:
            w.docker_fstype = docker_fstype
        # w.task_failure_count = task_failure_count
        w._is_master = is_master
        w._authorized = authorized
        w._connected = True
        return w

    def on_node_leave(self, host: Host):
        hostname = host.hostname
        w = self.workers.pop(hostname, None)
        if not w:
            return
        w._connected = False
        w.components = {}
        # NOTE 2021/1/18 none-stopped run implemented
    # Resource related

    def resume_resources(self, uid, resources, host):
        if not uid or not resources:
            return
        worker = self.get_worker(host)
        if not worker:
            return
        worker.reallocate(uid, resources)

    def check_available(self, flavor, host=None, exclude_host=None, exclude_allocated_tasks=[]):
        resources = flavor_get_resources(flavor)
        if not (set(resources) - {'local', 'hostname'}):
            return True
        # task require resource base on flavor
        require_gpu = resources.get('gpu', 0)
        require_mem = resources.get('memory', 0)
        require_cpu = resources.get('cpu', 0)
        require_gpu_type = resources.get('gpu_type', "any")
        require_tags = resources.get('tags') or []
        host_list = []
        if host:
            w = self.get_ready_worker(host)
            if w:
                host_list = [(host, w)]
        else:
            host_list = self.list_ready_workers()
        for hostname, worker in host_list:
            if exclude_host and hostname == exclude_host:
                continue
            if host and host != hostname:
                continue

            if not worker.check_tags(flavor_tags=require_tags):
                continue
            try:
                worker.check('', require_cpu, require_mem, require_gpu, require_gpu_type,
                             exclude_allocated_tasks=exclude_allocated_tasks)
                return True
            except Exception:
                continue
        return False

    def release_resources(self, uid):
        """
        Release resources previously reserved for a task
        """
        # release resources
        for _, worker in self.workers.items():
            # NOTE don't check alive, since in this step resources already released
            worker.deallocate(uid)

    def list_resource_spec(self, rtype):
        resources = {}
        for hostname, worker in self.list_ready_workers():
            resource = worker.resource_spec(rtype)
            resources.update({hostname: resource})
        return resources

    def get_gpu_resource_spec(self, gpu_identity):
        for _, worker in self.list_ready_workers():
            gpus = worker.resource_spec("gpu")
            for gpu in gpus:
                if gpu['uuid'] == gpu_identity:
                    return gpu['name']
        return gpu_identity

    def update_gpu_resource_spec(self, host: Host, resources: dict):
        hostname = host.hostname
        worker: Worker = self.workers.get(hostname, None)
        if not worker:
            return

        gpus = GpuResource()
        for gpu in resources['gpus']:
            categories = gpu.get('categories', '')
            gpus.add(
                gpu['uuid'],
                gpu['index'],
                gpu['name'],
                gpu['memory'],
                mig=gpu.get('mig', False),
                device_uuid=gpu.get('device_uuid'),
                instance_id=gpu.get('instance_id'),
                disable=gpu.get('disable', False),
                slice_instance=gpu.get('slice_instance', False),
                capacity=gpu.get('capacity', 1),
            )
            categories = gpu.get('categories', '')
            GPU_NAME_CATEGORY[gpu['name']] = categories
        # reload disabled GPU from db
        blacklist = json.loads(host.resources_blacklist) if host.resources_blacklist else {}
        gpu_blist = blacklist.get('gpu', [])
        for guid in gpu_blist:
            gpus.update(guid, disable=True)
        worker.update_gpu_resource(gpus)

    # def is_gpu_inuse(self, host: Host, gpu_uuid):
    #     worker: Worker = self.workers.get(host.hostname, None)
    #     if worker:
    #         return worker.worker_gpus.is_inuse(gpu_uuid)
    #     return False

    def is_gpu_any_inuse(self, host: Host):
        worker: Worker = self.workers.get(host.hostname, None)
        if worker:
            return worker.worker_gpus.any_inuse()
        return False

    def list_resources(self) -> dict:
        resources = {}
        for hostname, worker in self.list_ready_workers():
            resource = worker.resource_list()
            if resource:
                resources.update({hostname: resource})
        summary = self.resources_summary()
        resources.update({"total": summary})
        return resources

    def list_heartbeats(self, period=0):
        result = {}
        current = datetime.utcnow()
        for hostname, worker in self.list_ready_workers():
            if period:
                if worker.get_heartbeat_now:
                    continue
                if current - worker._last_heartbeat > timedelta(seconds=period):
                    continue
            result.update({hostname: worker.tasks_heartbeat})
        return result

    def find_resource_iterate(self, requested, task_uuid=None) -> tuple[str, dict]:
        out_of_gpu = 0
        # task require resource base on flavor
        require_tags = requested.get('tags') or []
        for hostname, worker in self.sorted_scoring_workers(requested, task_uuid):
            try:
                if not worker.check_tags(flavor_tags=require_tags):
                    continue
                out_of_gpu += 1
                resources_found = worker.find_resource(requested, task_uuid)
                return hostname, resources_found
            except Exception as e:
                if "Out of GPU resources" not in str(e):
                    out_of_gpu -= 1
                continue
        if out_of_gpu > 0:
            raise OutOfResource("Out of GPU resources", device="GPU")
        else:
            raise OutOfResource("Out of resources, please try later")

    def resources_maximum(self) -> dict:
        """Maximum remaining resource on workers

        NOTE: The maximum resource combination may not be satisfied on a single worker.
        """
        cpu = 0
        gpu = {'any': 0}
        memory = 0
        for _, worker in self.list_ready_workers():
            resource = worker.resource_list()
            for k, v in resource.items():
                if k not in ['cpu', 'gpu', 'memory']:
                    continue
                if k == 'gpu':
                    for gpu_name, gpu_val in v['remaining'].items():
                        if gpu_val > gpu.get(gpu_name, 0):
                            gpu[gpu_name] = gpu_val
                else:
                    v = v['remaining']
                    if k == 'cpu':
                        if v > cpu:
                            cpu = v
                    elif k == 'memory':
                        if v > memory:
                            memory = v

        return {
            'cpu': cpu,
            'gpu': gpu,
            'memory': memory
        }

    def resources_remain(self,
                         override_gpu_disable: Optional[Dict[str, bool]] = None,
                         override_gpu_mig: Optional[Dict[str, int]] = None) -> dict:
        cpu = 0
        gpu = {'any': 0}
        memory = 0
        for _, worker in self.list_ready_workers():
            resource = worker.resource_list(override_gpu_disable=override_gpu_disable,
                                            override_gpu_mig=override_gpu_mig)
            for k, v in resource.items():
                if k == 'cpu':
                    cpu += v.get('remaining', 0)
                elif k == 'gpu':
                    for gpu_name, gpu_val in v['remaining'].items():
                        gpu[gpu_name] = gpu.get(gpu_name, 0) + gpu_val
                elif k == 'memory':
                    memory += v.get('remaining', 0)

        return {
            'cpu': cpu,
            'gpu': gpu,
            'memory': memory
        }

    def resources_total(self) -> dict:
        cpu = 0
        gpu = 0
        memory = 0
        for _, worker in self.list_ready_workers():
            resource = worker.resource_list()
            for k, v in resource.items():
                if k == 'cpu':
                    cpu += v.get('total', 0)
                elif k == 'gpu':
                    gpu += v['total']['any']
                elif k == 'memory':
                    memory += v.get('total', 0)

        return {
            'cpu': cpu,
            'gpu': gpu,
            'memory': memory
        }

    def resources_summary(self) -> dict:
        preserved, p_used = self.resource_preserved()
        total = self.resources_total()
        remain = self.resources_remain()

        device = ["cpu", 'gpu', 'memory']
        unit = dict(zip(device, ["core", "", "mb"]))
        resources = {}

        for dt in device:
            resources[dt] = {}
            if unit[dt]:
                resources[dt].update({"unit": unit[dt]})
            dt_remain = remain[dt]
            dt_preserved = preserved[dt]
            dt_used = p_used[dt]
            if dt == 'gpu':
                dt_remain, dt_preserved, dt_used = (
                    x['any'] for x in (dt_remain, dt_preserved, dt_used)
                )
            resources[dt].update({
                "total": total[dt],
                "remain": dt_remain,
                "allocated": total[dt] - dt_remain,
                "reserved": {
                    "total": dt_preserved,
                    "allocated": dt_used,
                    "remain": dt_preserved - dt_used
                }
            })
        return resources

    def resource_preserved(self, override_plans: List[Union[Plan, PlanCache]] = []):
        """
        Calculate preserved and used resources for all users with preserved plans

        Args:
          override_plans: changes of plans that have not been written to DB/cache
        """
        from mlsteam.core.tasks.helper import TaskOperator
        device_types = ['cpu', 'gpu', 'memory']
        preserved = {'cpu': 0, 'gpu': {'any': 0}, 'memory': 0}
        used = {'cpu': 0, 'gpu': {'any': 0}, 'memory': 0}

        preserved_plans = plan_list(preserved=True)  # cached
        if override_plans:
            # cache.preserved  override.preserved  preserved_plans
            # ---------------  ------------------  ---------------
            # don't care       N/A                 no-op
            # N/A              True                add
            # N/A              False               no-op
            # True             True                replace
            # True             False               delete
            # False (impossible)
            preserved_plans_new = []
            override_plans_map = {p.id: p for p in override_plans}
            for p in preserved_plans:
                if p.id not in override_plans_map:
                    preserved_plans_new.append(p)
                elif p.preserved:
                    if override_plans_map[p.id].preserved:
                        preserved_plans_new.append(override_plans_map[p.id])
            for p_override in override_plans:
                if p_override.preserved and not any((p_override.id == p_new.id for p_new in preserved_plans_new)):
                    preserved_plans_new.append(p_override)
            preserved_plans = preserved_plans_new

        users: Iterable[User] = User.query.filter(User.plan.in_([p.id for p in preserved_plans]))  # DB read
        for u in users:
            user_preserved_plan = u.plan_obj  # cached
            if override_plans and user_preserved_plan.id in override_plans_map:
                user_preserved_plan = override_plans_map[user_preserved_plan.id]
            user_used = TaskOperator.task_resource_occupied(u.username)  # cached
            for dt in device_types:
                if dt == 'gpu':
                    for gpu_val, gpu_name in user_preserved_plan.gpu:
                        gpu_name = norm_gpu_type(gpu_name)
                        preserved['gpu'][gpu_name] = preserved['gpu'].get(gpu_name, 0) + (
                            gpu_val if gpu_val > 0 else user_used['gpu'].get(gpu_name, 0)
                        )
                    for gpu_name, gpu_val in user_used['gpu'].items():
                        gpu_name = norm_gpu_type(gpu_name)
                        used['gpu'][gpu_name] = used['gpu'].get(gpu_name, 0) + gpu_val
                else:
                    # avoid strange number if empty plan or plan have changed (also replace unlimited by used)
                    preserved[dt] += (max(getattr(user_preserved_plan, dt), 0) or user_used.get(dt, 0))
                    used[dt] += user_used.get(dt, 0)
        return preserved, used

    def check_preserve(self, preserve_attempt={}, allocate_attempt={},
                       occupied_before_preseve={}, use_preserved=False,
                       override_plans: List[Union[Plan, PlanCache]] = [],
                       override_gpu_disable: Optional[Dict[str, bool]] = None,
                       override_gpu_mig: Optional[Dict[str, int]] = None):
        r_remain = self.resources_remain(override_gpu_disable=override_gpu_disable,
                                         override_gpu_mig=override_gpu_mig)
        r_preserved, r_preserve_used = self.resource_preserved(override_plans=override_plans)
        for res in ['cpu', 'gpu', 'memory']:
            if res == 'gpu':
                p_attempt = {norm_gpu_type(gpu_name): val for gpu_name, val in preserve_attempt.get('gpu', {}).items()}
                a_attempt = {norm_gpu_type(gpu_name): val for gpu_name, val in allocate_attempt.get('gpu', {}).items()}
                occupied = occupied_before_preseve.get('gpu', {})
            else:
                p_attempt = preserve_attempt.get(res, 0)
                a_attempt = allocate_attempt.get(res, 0)
                occupied = occupied_before_preseve.get(res, 0)
            # |   PU   |   P-PU        |   R-(P-PU)           |
            # |   PU+o  |   P+p-(PU+o)   |   R-(P+p-(PU+o))   |
            # |  PU+o+a   |   P+p-(PU+o+a) | R-(P+p-(PU+o+a)) | if use_preserved
            if res == 'gpu':
                for gpu_name in r_remain['gpu'].keys():
                    remain = r_remain['gpu'][gpu_name] - a_attempt.get(gpu_name, 0)
                    preserved = r_preserved['gpu'].get(gpu_name, 0) + max(p_attempt.get(gpu_name, 0), 0)
                    preserved_used = r_preserve_used['gpu'].get(gpu_name, 0) + occupied.get(gpu_name, 0)
                    if use_preserved:
                        preserved_used += a_attempt.get(gpu_name, 0)

                    if remain < preserved - preserved_used or remain < 0:
                        if getattr(self, '_check_preserve_gpu_debug_log_time', 0) + 300 < (curr_time := time.time()):
                            self._check_preserve_gpu_debug_log_time = curr_time
                            logger.warning('gpu %s: remain=%d, preserved=%d, preserved_used=%d',
                                           gpu_name, remain, preserved, preserved_used)
                        raise OutOfResource(f'Out of {res.upper()} resource', device=res)
            else:
                remain = r_remain[res] - a_attempt
                preserved = r_preserved[res] + max(p_attempt, 0)
                preserved_used = r_preserve_used[res] + occupied
                if use_preserved:
                    preserved_used += a_attempt

                if remain < preserved - preserved_used or remain < 0:
                    # msg = 'Out of {} resource , remain:{}, preserve:{}, used:{}, p_attempt:{}, a_attempt:{}'.format(
                    #     res,
                    #     r_remain,
                    #     r_preserved,
                    #     r_preserve_used,
                    #     preserve_attempt,
                    #     allocate_attempt
                    # )
                    # logger.debug(msg)
                    raise OutOfResource(
                        f"Out of {res.upper()} resource", device=res)

    def rocm_available(self):
        resources = self.list_resource_spec('gpu')
        for _, res in resources.items():
            for gpu in res:
                if gpu['uuid'].startswith("APU"):
                    return True
        return False


worker_resources = WorkerResources()
