import pytz
import gevent
import time
import itertools

from datetime import datetime
from mlsteam.log import logger
from mlsteam.core.utils.image import split_image_domain
from mlsteam.core.container.dockerimage import ImageThread
from mlsteam.core.container.dockerimage import ImageM
from mlsteam.exception.base import HarborException, ImageException
from mlsteam.core.container.registry.harbor import HarborAPI
from mlsteam.core.models.run import RunState


# exception handler
def harbor_exception(func):
    def inner_function(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise HarborException(e) from e
    return inner_function


class HarborThread(ImageThread):
    def __init__(self, fqdn, port, account, token):
        super().__init__()
        self.domain = f"{fqdn}:{port}"
        self.auth = {'username': account, 'password': token}
        self.api = HarborAPI(fqdn, port, account, token)
        self.api.check_ssl()
        self.check_counter = itertools.cycle([1] + 59 * [0])
        self.init_gc()

    @property
    def check_flag(self):
        return next(self.check_counter)

    # call by run_task, query for all images on harbor
    def image_cache_load_thread(self, namespace_scan_scope):
        if self.check_flag:
            try:
                self.api.check_ssl(verb=False)
            except Exception as e:
                logger.error("[HarborThread]: check url error: {}".format(e))
                return []

        new_scan_image_list = []
        images_notag = []
        if not namespace_scan_scope:
            namespace_scan_scope = [p['name'] for p in self.api.list_project()]
        for ns in namespace_scan_scope:
            try:
                images_notag.extend(self.api.list_repository(projectname=ns))
            except Exception as e:
                print(str(e))
        for img in images_notag:
            # Optimize thread termination
            if self.stop_event.is_set():
                break
            self.image_load_repo(img, new_scan_image_list)
            # img:
            # {'artifact_count': 1,
            #  'id': 1,
            #  'name': 'test/python-gpu',
            #  'project_id': 2,
            #  'pull_count': 0,
            #  'creation_time': '2021-12-23T03:09:08.474Z',
            #  'update_time': '2021-12-23T03:09:08.474Z',
            # }
        return new_scan_image_list

    def image_load_repo(self, img, new_scan_image_list):
        ns, rep = img["name"].split("/", 1)
        arts = self.api.list_artifacts(projectname=ns, repositoryname=rep)
        for art in arts:
            tags = art.get("tags")
            if not tags:
                continue
            art_digest = art.get("digest")
            try:
                art_histories = self.api.get_artifacts_histories(
                    projectname=ns, repositoryname=rep, reference=art_digest
                )
                art_layers = len([i for i in art_histories if not i.get("empty_layer")])
            except Exception:
                art_layers = 0
            for tag in tags:
                try:
                    reg_name = self.api.addr
                    full_name = img["name"]
                    tag_name = tag["name"]
                    imgname = f"{reg_name}/{full_name}:{tag_name}"
                    image = ImageM(imgname)
                    image.uuid = art["digest"][7:17]
                    image.layer = art_layers
                    image.set_size_bytes(art["size"])
                    image.architecture = art["extra_attrs"].get(
                        "architecture", image.architecture
                    )
                    image.labels = art["extra_attrs"]["config"].get(
                        "Labels", image.labels
                    )
                    image.working_dir = art["extra_attrs"]["config"].get(
                        "WorkingDir", image.working_dir
                    )
                    image.exposed_ports = art["extra_attrs"]["config"].get(
                        "ExposedPorts", image.exposed_ports
                    )
                    image.env = art["extra_attrs"]["config"].get("Env", image.env)
                    scan_overview = art.get("scan_overview")
                    # scan_overview:
                    # "application/vnd.security.vulnerability.report; version=1.1": {
                    #     "report_id": "8251d5ac-948c-4b04-b2cf-90fd4dbe77f6",
                    #     "start_time": "2021-12-29T02:15:38.000Z",
                    #     "end_time": "2021-12-29T02:15:52.000Z",
                    #     "duration": 14,
                    #     "complete_percent": 100,
                    #     "scan_status": "Success",
                    #     "severity": "Critical",
                    #     "scanner": {
                    #         "name": "Trivy",
                    #         "vendor": "Aqua Security",
                    #         "version": "v0.20.1"
                    #     },
                    #     "summary": {
                    #         "fixable": 477,
                    #         "summary": {
                    #         "Critical": 7,
                    #         "High": 118,
                    #         "Low": 281,
                    #         "Medium": 500,
                    #         "Unknown": 9
                    #         },
                    #         "total": 915
                    #     }
                    # }
                    if scan_overview:
                        scan_overview = next(
                            iter(scan_overview.values())
                        )  # get first item
                        image.scan_data["status"] = scan_overview.get("scan_status")
                        image.scan_data["severity"] = scan_overview.get("severity")
                        image.scan_data["summary"] = scan_overview.get("summary")
                        image.scan_data["url"] = (
                            "{}/harbor/projects/{}/repositories/{}/artifacts/{}".format(
                                self.domain, art["project_id"], rep, art["digest"]
                            )
                        )
                    try:
                        created = art["extra_attrs"][
                            "created"
                        ]  # E.g., "2022-07-01T04:41:29.440112514Z"
                        image.created_time = pytz.utc.localize(
                            datetime.strptime(
                                created.split(".")[0], "%Y-%m-%dT%H:%M:%S"
                            )
                        )
                    except Exception:
                        pass
                    self.image_cache_add(image)
                    new_scan_image_list.append(imgname)
                    self.image_current += 1
                except Exception:
                    continue

    # call by API
    # warn: server will delete same uuid tags while deleting one of image tags
    @harbor_exception
    def image_delete(self, imagename, force=False):  # pylint: disable=arguments-renamed
        # ex. 'docker.io', 'library/hello-world', 'latest'
        _, image = self.image_cache_get(imagename)
        if image:
            _, img, tag = split_image_domain(imagename)
            if '/' in img:
                ns, img = img.split('/', 1)  # a/b/c -> a, b/c
                if ns and img:
                    # delete from server
                    self.api.delete_artifact(projectname=ns, repositoryname=img, tagname=tag)
                    return
        logger.warning("Delete image {} not found".format(imagename))

    def _image_push_summary(self, image_layers):
        summary = {}
        for _, status in image_layers.items():
            if not isinstance(status, str):
                continue
            if summary.get(status):
                summary[status] += 1
            else:
                summary[status] = 1
        total_steps = len(image_layers) * 4
        now_steps = (
            summary.get("Preparing", 0) * 1 +
            summary.get("Waiting", 0) * 2 +
            summary.get("Pushing", 0) * 3 +
            summary.get("Pushed", 0) * 4 +
            summary.get("Layer already exists", 0) * 4 +
            summary.get("Mounted from", 0) * 4
        )
        return {
            "steps": f"{now_steps}/{total_steps}",
            "layers": summary
        }

    @harbor_exception
    def image_push(self, imagename, task=None):
        from mlsteam.core.container import docker_apiclient_timeout
        from mlsteam.core.utils.image import DockerPushSummary
        domain, image, tag = split_image_domain(imagename)
        imagerepo = f"{domain}/{image}"
        cli = docker_apiclient_timeout()
        # push image to repository
        push_start_t = time.time()
        summary = DockerPushSummary(f"{imagerepo}:{tag}")
        for status in cli.push(imagerepo, tag, stream=True, auth_config=self.auth, decode=True):
            if status.get('errorDetail'):
                err = status.get('errorDetail')
                raise Exception(f"Push image {imagename} failed, {err.get('message')}")
            if self.stop_event.is_set():
                raise Exception(f"Pushing image {imagename} Interrupted")
            if task and task.status != RunState.RUN:
                raise Exception(f"Task {task.uuid} Interrupted")
            # self._image_push_status(image_layers, status)
            summary.step_update(status)
            if task:
                task.message = "pushing to local registry: {}".format(summary.layers_summary())
                # send browser socketio, no db write
                task.update_status(task.status)
                # task.attrs = {"percentage": self._image_push_percent(image_layers)}
            gevent.sleep(0.5)
        # self.send_progress(100.00, time.time() - push_start_t)
        total_time = time.time() - push_start_t
        logger.info("Harbor image push took {:.2f} seconds".format(total_time))

    @harbor_exception
    def user_create(self, username):
        if not self.api.get_userid(username):
            self.api.create_user(username)

    @harbor_exception
    def user_delete(self, username):
        uid = self.api.get_userid(username)
        logger.info("Harbor delete user {}({})".format(username, uid))
        if uid:
            self.api.delete_user(uid)

    @harbor_exception
    def user_admin_permission(self, username, permission: bool):
        uid = self.api.get_userid(username)
        logger.info("Harbor set user {}({}) admin permission = {}".format(username, uid, permission))
        if uid:
            self.api.set_user_admin_permission(userid=uid, permission=permission)

    @harbor_exception
    def group_create(self, namespace, username=None):
        '''username will be default member'''
        from mlsteam.core.models.project import ProjectMemberPermission
        p = self.api.get_project(projectname=namespace)
        if not p:
            self.api.create_project(projectname=namespace)
            if username:
                logger.info("Harbor creates project {} by {}".format(namespace, username))
                self.group_member_add(
                    groupname=namespace,
                    username=username,
                    permission=ProjectMemberPermission.OWNER
                )

    @harbor_exception
    def group_delete(self, namespace):
        p = self.api.get_project(projectname=namespace)
        if not p:
            return
        reps = self.api.list_repository(projectname=namespace)
        for rep in reps:
            rep_name = next(reversed(rep['name'].split('/', 1)))
            self.api.delete_repository(projectname=namespace, repositoryname=rep_name)
        images = self.image_cache_list([namespace])
        for image in images:
            self.image_cache_delete(image.name)
        self.api.delete_project(projectname=namespace)
        logger.info("Harbor remove project {}".format(namespace))

    @harbor_exception
    def group_member_add(self, groupname, username, permission):
        '''https://goharbor.io/docs/latest/administration/managing-users/user-permissions-by-role/'''
        from mlsteam.core.models.project import ProjectMemberPermission
        # 1 Project Admin
        # 2 Developer
        # 3 Guest
        # 4 Maintainer
        # 5 Limited Guest
        roleid = 3
        if permission == ProjectMemberPermission.OWNER:
            roleid = 1
        elif permission in ProjectMemberPermission.PERM_WRITE:
            roleid = 2
        mb = self.api.get_member(projectname=groupname, username=username)
        if mb:
            if mb['role_id'] != roleid:
                self.api.update_member_role(projectname=groupname, mbid=mb['id'], roleid=roleid)
                logger.info("Harbor project {} update member {} as {}".format(groupname, username, permission))
        else:
            self.api.join_project(projectname=groupname, username=username, roleid=roleid)
            logger.info("Harbor project {} add member {} as {}".format(groupname, username, permission))

    @harbor_exception
    def group_member_delete(self, groupname, username):
        mbid = self.api.get_memberid(projectname=groupname, username=username)
        self.api.leave_project(projectname=groupname, mbid=mbid)
        logger.info("Harbor deletes project {} member {}".format(groupname, username))

    @harbor_exception
    def image_tag(self, imagename, new):
        # 會將原有tag覆蓋，如果不想覆蓋請在更前端檢查
        _, to_image, newtag = split_image_domain(new)
        _, from_image, tag = split_image_domain(imagename)
        if '/' not in to_image:
            raise ImageException('image format must be <namespace>/<image>')
        proj, repo = to_image.split('/', 1)
        self.group_create(proj)
        if from_image == to_image:
            # if tag == newtag, skip it
            if newtag != tag:
                self.api.delete_tag(projectname=proj, repositoryname=repo, tagname=newtag)
                self.api.create_tag(projectname=proj, repositoryname=repo, tagname=tag, newtag=newtag)
        else:
            # 因copy會將所有tags，無條件覆蓋目標tag，造成非預期結果
            # copy之前必須製造一個中轉目標，存到中轉處之後修剪其tag到剩一個，才將其copy到目標。
            tmp_suffix = '-system-copy-tmp'  # 中轉用
            # src -> tmp
            self.api.copy_artifacts(projectname=proj, repositoryname=repo+tmp_suffix, from_image=f'{from_image}:{tag}')
            try:
                # existed tags
                existed_tags = self.api.list_tags(projectname=proj, repositoryname=repo+tmp_suffix, tagname=tag)
                # pick the name
                existed_tags = [t['name'] for t in existed_tags]
                # if new tag not in tags, create one
                if newtag not in existed_tags:
                    self.api.create_tag(projectname=proj, repositoryname=repo+tmp_suffix, tagname=tag, newtag=newtag)
                else:
                    # delete others except newtag
                    existed_tags.remove(newtag)
                # remove unnecessary tags
                for t in existed_tags:
                    self.api.delete_tag(projectname=proj, repositoryname=repo+tmp_suffix, tagname=t)
                # tmp with "only one tag" copy to target
                self.api.copy_artifacts(projectname=proj, repositoryname=repo,
                                        from_image=f'{proj}/{repo}{tmp_suffix}:{newtag}')
            except Exception as e:
                raise e
            finally:
                # clean tmp
                try:
                    self.api.delete_repository(projectname=proj, repositoryname=repo+tmp_suffix)
                except Exception:
                    pass

    @harbor_exception
    def image_vulnerabilities_scan(self, imagename=None):
        if imagename:
            _, image, tag = split_image_domain(imagename)
            if '/' not in image:
                raise ImageException('cannot parse repository')
            proj, repo = image.split('/', 1)
            self.api.scan_image(projectname=proj, repositoryname=repo, tagname=tag)
            _, imageM = self.image_cache_get(imagename)
            if imageM:
                imageM.scan_data['status'] = 'Running'
        else:
            self.api.scan_now()  # scan all image
            for imageM in self._images_cache_list:
                imageM.scan_data['status'] = 'Running'

    @harbor_exception
    def image_vulnerabilities_progress(self):
        return self.api.get_scan_metrics()

    def init_gc(self):
        try:
            if self.api.get_gc_schedule():
                return
            self.api.set_gc_schedule(cron='hourly')
        except Exception as e:
            logger.warning("Harbor init GC failed: {}".format(e))


# tests
if __name__ == '__main__':
    # blueprint error in unit test
    reg = HarborThread("192.168.0.17", "10394", "admin", "harboradmin")
    reg.start()
    for i in range(50):
        print(len(reg.image_cache_list()))
        gevent.sleep(1)
    reg.stop()
