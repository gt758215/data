import base64
import hashlib
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Dict, Optional, Union

from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from ruamel.yaml import YAML
from sqlalchemy.orm.attributes import flag_modified
from tenacity import retry, stop_after_attempt, wait_exponential

from mlsteam.core import nginx
from mlsteam.core.const import CLUSTER_CRED_ROOT, K8S_INSTANCE_LABEL
from mlsteam.core.kubernetes import KubernetesTool
from mlsteam.core.models import Host
from mlsteam.core.models.cluster import (Cluster, ClusterHosts, ClusterRole,
                                         ClusterType)
from mlsteam.core.models.run import RunStateGroup
from mlsteam.core.redis import RedisCache
from mlsteam.core.tasks.helper import TaskOperator
from mlsteam.core.utils.dbsession import commit_on_close
from mlsteam.core.utils.verifier import _VALID_USERNAME_REGEX, between_length
from mlsteam.deployment import DEPLOYMENT_SRC_ROOT
from mlsteam.exception.base import ClusterException, HostException
from mlsteam.exception.cluster import ClusterNotFound
from mlsteam.exception.datastore import DatastoreNotFound
from mlsteam.exception.host import HostAuthoried
from mlsteam.exception.request import DataVerifyError
from mlsteam.initapp import app
from mlsteam.log import logger
from mlsteam.utils import parse_k8s_size, update_yaml_file
from mlsteam.webapp import db


class BaseCluster(object):
    """ Default inteface of Cluster """

    CLUSTER_LOCK = RLock()

    def __init__(self, **kwargs):
        self.cluster_id = kwargs.pop('cluster', None)
        self.cluster_uuid = kwargs.pop('cluster_uuid', None)
        self.cluster_type = None

    def init(self, **kwargs):
        raise NotImplementedError()

    def set_cluster_id(self, cluster_id):
        self.cluster_id = cluster_id

    def update_cluster(self, **kwargs):
        raise NotImplementedError()

    def teardown_cluster(self, **kwargs):
        raise NotImplementedError()

    def node_join(self, host, authorize):
        raise NotImplementedError()

    def node_leave(self, cluster_node_id):
        raise NotImplementedError()

    def node_list(self):
        return ClusterHosts.query.filter(ClusterHosts.cluster_id == self.cluster_id).all()

    @property
    def has_agent(self):
        return True

    @property
    def cluster(self):
        return Cluster.query.filter(Cluster.id == self.cluster_id).first()

    @property
    def serialize(self):
        return self.cluster.serialize


class KubernetesCluster(BaseCluster):
    """Kubernetes cluster"""

    CREDS = {
        'server_ca': 'ca.crt',
        'client_cert': 'client.crt',
        'client_key': 'client.key'
    }

    def __init__(self, cluster_configs: Optional[dict] = None, **kwargs):
        super().__init__(**kwargs)
        self.cluster_type = ClusterType.KUBERNETES
        self.cluster_configs = cluster_configs or {}
        self.all_node_ready = False

    def init(self, **kwargs):
        if 'update_configs' in kwargs:
            self.save_credentials(update_configs=kwargs['update_configs'])  # validate & save
            self.cluster_configs.update(kwargs['update_configs'])
        nginx.reset(clean_up=False)
        self.setup_clusterpolicy()
        self.setup_term_service()
        self.setup_nodes()

    def setup_clusterpolicy(self):
        self.k8s_tool.create_mig_configmap()
        self.k8s_tool.patch_default_mig_config()

    def teardown_cluster(self, **kwargs):
        from mlsteam.core.host import host_delete, host_offline
        force = kwargs.get('force', False)

        if not (cluster := Cluster.query.filter_by(id=self.cluster_id).first()):
            raise ClusterNotFound(f'Cluster {self.cluster_id} not found', str(self.cluster_id))
        if self.is_occupied_by_tasks():
            if not force:
                raise ClusterException(f'Cannot delete cluster {self.cluster_id}'
                                       ' since there are tasks occupying the hosts')
            # NOTE: attempt to delete tasks
        with commit_on_close():
            all_c_hosts = [{'id': c_host.id, 'host/id': c_host.host.id, 'host/host_id': c_host.host.host_id}
                           for c_host in cluster.cluster_hosts]
        for c_host in all_c_hosts:
            host_offline(c_host['host/host_id'])
            self.node_leave(c_host['id'])
            host_delete(c_host['host/id'], db_delete=True)

        self.teardown_term_service()
        self.delete_credentials()

    def update_cluster(self, **kwargs):
        updates = {}
        try:
            if not (cluster := Cluster.query.filter_by(id=self.cluster_id).first()):
                raise ClusterNotFound(f'Cluster {self.cluster_id} not found', str(self.cluster_id))
            if (name := kwargs.pop('cluster_name', None)):
                updates['name'] = name
                cluster.name = name
            if (configs := kwargs.pop('cluster_configs', None)):
                updates['configs'] = configs
                self.save_credentials(update_configs=configs)  # validate & save
                self.cluster_configs.update(configs)
                cluster.configs.update(configs)
                flag_modified(cluster, 'configs')
            nginx.reset(clean_up=False)
            self.setup_term_service()
            self.setup_nodes()
            db.session.add(cluster)
            db.session.commit()
            logger.info('Update cluster %d success, updates=%s', self.cluster_id, updates)
        except Exception:
            db.session.rollback()
            raise

    def node_join(self, host, authorize):
        try:
            if ClusterHosts.query.filter_by(host_id=host.id).first():
                raise HostAuthoried(f"Host {host.hostname} already in cluster", host=host.hostname)
            cluster_node = ClusterHosts(
                cluster_id=self.cluster_id,
                host_id=host.id,
                cluster_role=ClusterRole.WORKER  # we do not distinguish between K8s masters and workers
            )
            db.session.add(cluster_node)
            db.session.commit()
            logger.info('Node {} join into {} success'.format(host, self.__class__.__name__))
        except Exception:
            db.session.rollback()
            import traceback
            logger.error('Node {} join into {} fail, {}'.format(
                host, self.__class__.__name__, traceback.format_exc()))
            raise

    def node_leave(self, cluster_node_id):
        node = None
        try:
            if not (c_host := ClusterHosts.query.filter_by(id=cluster_node_id).first()):
                return
            db.session.delete(c_host)
            db.session.commit()
            logger.info('Node {} leave {} success'.format(c_host, self.__class__.__name__))
        except Exception as e:
            db.session.rollback()
            logger.error('Node {} leave {} fail, {}'.format(node, self.__class__.__name__, str(e)))
            raise e

    @property
    def has_agent(self):
        return False

    @property
    def cred_dir(self):
        return Path(CLUSTER_CRED_ROOT) / str(self.cluster_uuid)

    @property
    def _term_service_uuid(self):
        return f'terms-{self.cluster_uuid}'

    @property
    def k8s_tool(self):
        return KubernetesTool(self.cluster_uuid, self.cluster_configs['server_addr'])

    def save_credentials(self, update_configs: dict):
        cred_dir = self.cred_dir
        cred_dir.mkdir(parents=True, exist_ok=True)
        for cred_key, cred_file in self.CREDS.items():
            if cred_key in update_configs:
                if (cred_val := update_configs[cred_key]):
                    self.load_validate_credential(cred_key, cred_val)
                    with (cred_dir / cred_file).open('wt', encoding='utf-8') as fp:
                        fp.write(cred_val)
                else:
                    (cred_dir / cred_file).unlink(missing_ok=True)
        kubecfg = {
            'apiVersion': 'v1',
            'kind': 'Config',
            'preferences': {},
            'clusters': [{
                'name': 'cluster',
                'cluster': {
                    'server': update_configs.get('server_addr', self.cluster_configs.get('server_addr')),
                    'certificate-authority-data': self._b64_enc(update_configs.get('server_ca'))
                }
            }],
            'users': [{
                'name': 'user',
                'user': {
                    'client-certificate-data': self._b64_enc(update_configs.get('client_cert')),
                    'client-key-data': self._b64_enc(update_configs.get('client_key'))
                }
            }],
            'contexts': [{
                'name': 'default',
                'context': {
                    'cluster': 'cluster',
                    'user': 'user'
                }
            }],
            'current-context': 'default'
        }
        with (cred_dir / 'config').open('wt', encoding='utf-8') as fp:
            yaml = YAML(typ='safe')
            yaml.dump(kubecfg, fp)

    @classmethod
    def _b64_enc(cls, str_data: Optional[str]) -> Optional[str]:
        if str_data:
            return base64.b64encode(str_data.encode('utf-8')).decode('utf-8')
        return None

    def delete_credentials(self):
        try:
            shutil.rmtree(self.cred_dir)
        except Exception:
            logger.exception('failed to delete credentials for cluster %s', self.cluster_uuid)

    @classmethod
    def load_validate_credential(cls, cred_key, cred_data: Union[str, bytes]) -> Optional[datetime]:
        if isinstance(cred_data, str):
            cred_data = cred_data.encode('utf-8')
        if cred_key in ('server_ca', 'client_cert'):
            cert = x509.load_pem_x509_certificate(cred_data, default_backend())
            return cert.not_valid_after.replace(tzinfo=timezone.utc)
        elif cred_key == 'client_key':
            serialization.load_pem_private_key(cred_data, password=None)
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(), reraise=True)
    def get_nodes(self):
        return self.k8s_tool.list_nodes()

    def get_node_hostid(self, machine_id):
        h = 'k8s-{}-{}'.format(self.cluster_uuid, machine_id)
        return hashlib.sha256(h.encode('utf-8')).hexdigest()[:16]

    def setup_nodes(self):
        from mlsteam.core.host import host_create, host_get_by_id, host_offline, host_online
        try:
            cluster_nodes = self.get_nodes()
        except Exception:
            with commit_on_close() as db_session:
                cluster_hostids = [c_host.host.host_id for c_host in
                                   db_session.query(ClusterHosts).filter_by(cluster_id=self.cluster_id).all()]
            for c_hostid in cluster_hostids:
                host_offline(c_hostid)
            raise
        ready_node_count = 0
        cluster_node_hostids = set()
        for node in cluster_nodes:
            host_id = self.get_node_hostid(node['status']['nodeInfo']['machineID'])

            cluster_node_hostids.add(host_id)
            hostname = node['metadata']['labels'].get('kubernetes.io/hostname') or node['metadata']['name']
            nvidia_labels = dict(filter(
                lambda i: i[0].startswith("nvidia.com/"),
                node['metadata']['labels'].items()
            ))
            deploying_validator_state = nvidia_labels.get('nvidia.com/gpu.deploy.operator-validator', 'true')
            deploying_gfd_state = nvidia_labels.get('nvidia.com/gpu.deploy.gpu-feature-discovery', 'true')
            mig_capable = nvidia_labels.get('nvidia.com/mig.capable', '')
            mig_state = nvidia_labels.get('nvidia.com/mig.config.state', 'success')
            mig_config = nvidia_labels.get('nvidia.com/mig.config', '')
            gpu_mig_count = sum(
              [int(v) for v in dict(filter(
                lambda i: i[0].startswith("nvidia.com/"),
                node['status']['allocatable'].items())).values()]
            )

            # === Check GPU ready, set node to disconnected if not ready ===
            deploying = deploying_validator_state != 'true' or deploying_gfd_state != 'true'
            if not nvidia_labels:
                # cpu nodes
                pass
            elif not mig_capable:
                # deploying, have no label
                continue
            elif mig_capable == 'true':  # GPU with MIG
                if deploying or mig_state not in ['success', 'failed']:
                    logger.debug(
                        '[GPU with MIG] {} operator-validator: {}, gpu-feature-discovery: {}, '
                        'mig.config.state: {}, gpu_mig_count: {}'.format(
                            hostname,
                            deploying_validator_state,
                            deploying_gfd_state,
                            mig_state,
                            gpu_mig_count,
                        )
                    )
                    host_offline(host_id)
                    continue
            else:  # normal GPU or NO GPU
                if deploying:
                    logger.debug(
                        '[GPU] {} operator-validator: {} gpu-feature-discovery: {}'.format(
                            hostname,
                            deploying_validator_state,
                            deploying_gfd_state,
                        )
                    )
                    host_offline(host_id)
                    continue
            # ================================================================
            host_ip = next((a['address'] for a in node['status']['addresses'] if a['type'] == 'InternalIP'), None)
            resources = {
                'ip': host_ip,
                'hostname': hostname,
                'cpu_cores': parse_k8s_size(node['status']['allocatable']['cpu']),
                'memory_mb': parse_k8s_size(node['status']['allocatable']['memory']) / (1024 ** 2),
                'storage_mb': parse_k8s_size(node['status']['allocatable']['ephemeral-storage']) / (1024 ** 2),
                'addresses': {'ipv4': host_ip},
                'gpus': [],
                'roles': ['worker'],  # we do not distinguish between masters and slaves
                'specs': {},
                'nvidia.com': nvidia_labels,
                'nvidia_driver': nvidia_labels.get("nvidia.com/cuda.driver-version.full", ''),
                'nvidia_cuda': nvidia_labels.get("nvidia.com/cuda.runtime-version.full", ''),
                'nvidia_mig_capable': mig_capable,
                'videos': [],
                'platform': {
                    'machine': (node['metadata']['labels'].get('kubernetes.io/arch') or
                                node['status']['nodeInfo']['architecture']),
                    'system': (node['metadata']['labels'].get('beta.kubernetes.io/os') or
                               node['status']['nodeInfo']['operatingSystem']),
                }
            }

            if 'nvidia.com/gpu' in node['status']['allocatable']:
                if gpu_mig_count == 0:
                    # 0 gpu on host but it has nvidia.com labels
                    continue
                try:
                    gpu_index = 0
                    gpu_count = int(nvidia_labels['nvidia.com/gpu.count'])
                    gpu_name = nvidia_labels['nvidia.com/gpu.product']
                    gpu_memory = nvidia_labels['nvidia.com/gpu.memory']
                    if 'GB' not in gpu_name:
                        gpu_name = gpu_name + '-{}GB'.format(int(gpu_memory) // 1024)

                    # MIG ==>
                    mig_index = 0
                    mig_devices = []
                    for mig_key, mig_num in dict(filter(
                        lambda i: i[0].startswith("nvidia.com/mig-"),
                        node['status']['allocatable'].items()
                    )).items():
                        if int(mig_num) <= 0:
                            continue
                        mig_name = nvidia_labels[mig_key + '.product']
                        mig_memory = nvidia_labels[mig_key + '.memory']
                        mig_count = int(nvidia_labels[mig_key + '.count'])
                        for _ in range(mig_count):
                            mig_devices.append({
                                'index': str(mig_index + gpu_count),
                                'name': mig_name,
                                'uuid': f'NV-MIG-{mig_index}',
                                'memory': mig_memory,
                                'mig': False,
                                'categories': ['gpu.nvidia.mig'],
                            })
                            mig_index += 1
                    # <==
                    gpu_disable = len(mig_devices) > 0
                    for _ in range(gpu_count):
                        resources['gpus'].append({
                            'index': str(gpu_index),
                            'name': gpu_name,
                            'uuid': f'NV-GPU-{gpu_index}',
                            'memory': gpu_memory,
                            'mig': mig_capable,
                            'categories': ['gpu.nvidia'],
                            'disable': gpu_disable,
                            'mig_profile': mig_config,
                        })
                        gpu_index += 1
                    resources['gpus'].extend(mig_devices)
                except Exception:
                    import traceback
                    print(traceback.format_exc())
                    continue
            with self.CLUSTER_LOCK:
                host_create(host_id, hostname, resources, authorize=True, skip_host_record=True)
                host = host_get_by_id(host_id)
                if not host.in_cluster:
                    self.node_join(host, authorize=True)
                host_ready = next((c['status'].lower() == 'true'
                                   for c in node['status']['conditions'] if c['type'] == 'Ready'), False)
                if host_ready:
                    host_online(host_id, joined_ok=True, suppress_log=True,
                                k8s_allocatable=node['status']['allocatable'],
                                k8s_conditions=node['status']['conditions'])
                    ready_node_count += 1
                else:
                    host_offline(host_id)
        with commit_on_close() as db_session:
            cluster_hostids = [c_host.host.host_id for c_host in
                               db_session.query(ClusterHosts).filter_by(cluster_id=self.cluster_id).all()]
        for c_hostid in cluster_hostids:
            # hosts in db not listed by K8S server
            if c_hostid not in cluster_node_hostids:
                host_offline(c_hostid)
        self.all_node_ready = len(cluster_nodes) == ready_node_count

    def get_nodeport_ip(self) -> str | None:
        with RedisCache.open(f'manta.clusters.{self.cluster_uuid}.nodeport_ip') as cached:
            if not cached.exists:
                try:
                    for node in self.get_nodes():
                        host_ready = next((c['status'].lower() == 'true'
                                           for c in node['status']['conditions'] if c['type'] == 'Ready'), False)
                        if host_ready:
                            host_ip = next((a['address'] for a in node['status']['addresses']
                                            if a['type'] == 'InternalIP'), None)
                            cached.set(json.dumps(host_ip), ttl=300)
                            return host_ip
                    raise ClusterException(f'no active hosts for cluster {self.cluster_uuid}')
                except Exception:
                    cached.set(json.dumps(None), ttl=60)
                    return None
            else:
                return json.loads(cached.get())

    def setup_term_service(self):
        cred_dir = self.cred_dir
        shutil.copytree(DEPLOYMENT_SRC_ROOT / 'term_service', cred_dir, dirs_exist_ok=True)
        with update_yaml_file(cred_dir / 'kustomization.yaml') as cfg:
            cfg['namespace'] = app.config['MANTA_K8S_NAMESPACE']
            cfg['nameSuffix'] = f'-{self.cluster_uuid}'
            cfg['labels'][0]['pairs'][K8S_INSTANCE_LABEL] = str(self.cluster_uuid)
        with update_yaml_file(cred_dir / 'set_max_terms.yaml') as cfg:
            cfg[0]['value'] = '8'
        self.k8s_tool.apply_resources(cred_dir, app.config['MANTA_K8S_NAMESPACE'], kustomize=True)
        nginx.conf_create(
            nginx.RouteType.TERM_SERVICE, project_uuid=None, task_uuid=self._term_service_uuid,
            upstream_service=f'term-service-svc-{self.cluster_uuid}', cluster_uuid=str(self.cluster_uuid)
        )

    def teardown_term_service(self):
        try:
            nginx.conf_delete(self._term_service_uuid)
            self.k8s_tool.delete_resources(
                self.cred_dir, app.config['MANTA_K8S_NAMESPACE'], kustomize=True, ignore_not_found=True
            )
        except Exception:
            pass

    def is_occupied_by_tasks(self):
        tasks = TaskOperator.cache_query(not_status=RunStateGroup.OCCUPY_RSC)
        cluster_hosts = [h.host.hostname for h in ClusterHosts.query.filter_by(cluster_id=self.cluster_id).all()]
        hosts_in_cluster = [host for host in set([t.host for t in tasks]) if host in cluster_hosts]
        return bool(hosts_in_cluster)

    @property
    def serialize(self):
        ret = super().serialize
        for cert_key in ('server_ca', 'client_cert'):
            if (cert_data := ret['configs'].get(cert_key, None)):
                try:
                    cert = x509.load_pem_x509_certificate(cert_data.encode('utf-8'), default_backend())
                    ret['configs'][cert_key + '_expiry_time'] = cert.not_valid_after.replace(
                        tzinfo=timezone.utc).isoformat(timespec='seconds')
                except Exception as err:
                    ret['configs'][cert_key + '_error'] = str(err)
        return ret


class ClusterManager(object):
    def __init__(self):
        self.clusters: Dict[int, BaseCluster] = {}

    def init(self):
        with commit_on_close() as db_session:
            all_cinfo = [c.serialize for c in db_session.query(Cluster).all()]
        for c in all_cinfo:
            cid = c['id']
            try:
                self.clusters[cid] = self.create_cluster_instance(c['cluster_type'], cluster=cid,
                                                                  cluster_uuid=c['uuid'], cluster_configs=c['configs'])
            except Exception as e:
                logger.error('Cluster {} initialization fail dute to {}'.format(cid, e))
                continue
            try:
                self.clusters[cid].init()
            except Exception as e:
                logger.error('Cluster {} setup fail dute to {}'.format(cid, e))
                continue

    def create_cluster_instance(self, cluster_type, **kwargs):
        if cluster_type == ClusterType.SWARM:
            raise ClusterException(f"Create cluster failed, due to unsupport {cluster_type} type")
        elif cluster_type == ClusterType.KUBERNETES:
            return KubernetesCluster(**kwargs)
        else:
            raise ClusterException(f"Create cluster failed, due to unsupport {cluster_type} type")

    def create_cluster(self, cluster_name, cluster_type, cluster_uuid,
                       cluster_configs={}, node_list=[], **kwargs) -> int:
        try:
            self.validate_cluster_name(cluster_name)
            with BaseCluster.CLUSTER_LOCK:
                cluster_record = Cluster(name=cluster_name, cluster_type=cluster_type,
                                         uuid=cluster_uuid, configs=cluster_configs)
                db.session.add(cluster_record)
                db.session.flush()  # cluster_record.id is available now
                cluster = self.create_cluster_instance(cluster_type, cluster=cluster_record.id,
                                                       cluster_uuid=cluster_uuid, cluster_configs=cluster_configs)
                self.clusters[cluster.cluster_id] = cluster
                cluster.init(update_configs=cluster_configs)
                db.session.commit()
        except Exception as e:
            db.session.rollback()
            logger.error('Create Cluster fail due to {}'.format(str(e)))
            raise e

        self.clusters[cluster.cluster_id] = cluster

        # add node
        logger.info('Cluster created try to add nodes {}'.format(node_list))
        err = {}
        for host in node_list:
            try:
                cluster.node_join(host, host.authorize)
            except Exception as e:
                # When join fail, just ignore and show why fail to user.
                err[host.hostname] = str(e)
        if err:
            raise HostException(f"Add node to cluster {cluster_record} failed")

        return cluster.cluster_id

    def update_cluster(self, cluster_id, cluster_name=None, cluster_configs=None, **kwargs):
        try:
            if cluster_name:
                self.validate_cluster_name(cluster_name)
            c = self.clusters[cluster_id]
            c.update_cluster(cluster_name=cluster_name, cluster_configs=cluster_configs, **kwargs)
        except Exception as e:
            logger.exception('Cannot update cluster %d, update_args=%s', cluster_id,
                             {'name': cluster_name, 'configs': cluster_configs, **kwargs})
            raise ClusterException(f'Cannot update cluster {cluster_id}, {str(e)}') from e

    def list_cluster(self):
        clusters = []
        try:
            for c in self.clusters.values():
                data = c.serialize
                cluster_hosts = [n.serialize for n in c.node_list()]
                data.update({
                    'cluster_hosts': cluster_hosts
                })
                clusters.append(data)
            return clusters
        except (DatastoreNotFound, ClusterException):
            logger.exception('Cannot list clusters')
            return []

    def delete_cluster(self, cluster_id, **kwargs):
        try:
            c = self.clusters[cluster_id]
            c.teardown_cluster(**kwargs)
            Cluster.query.filter(Cluster.id == cluster_id).delete()
            db.session.commit()
            del self.clusters[c.cluster_id]
            logger.info('cluster delete {} success'.format(c))
        except Exception:
            db.session.rollback()
            logger.exception('cluster delete {} fail'.format(c))
            raise

    def get_cluster(self, cluster_id):
        if cluster_id not in self.clusters:
            return None
        return self.clusters[cluster_id]

    def get_default_k8s_cluster(self):
        k8s_clusters = [c for c in self.clusters.values() if isinstance(c, KubernetesCluster)]
        if not k8s_clusters:
            raise ClusterException('Kubernetes cluster not found')
        return k8s_clusters[0]

    def gen_cluster_uuid(self):
        _uuid = str(uuid.uuid4()).replace('-', '')[1:7]
        # ensure the uuid is unique
        while next((c for c in self.clusters.values() if c.cluster_uuid == _uuid), None):
            _uuid = str(uuid.uuid4()).replace('-', '')[1:7]
        return _uuid

    @classmethod
    def validate_cluster_name(cls, cluster_name):
        between_length(cluster_name, 3, 64, 'cluster name')
        if not _VALID_USERNAME_REGEX.match(cluster_name):
            raise DataVerifyError('Invalid cluster name: should start with an alphabet '
                                  'and contain only alphanumeerics, dots, and underscores', data='cluster name')


cluster_manager = ClusterManager()


def cluster_available_hosts():
    hosts_in_cluster = [x.host_id for x in db.session.query(ClusterHosts.host_id).distinct()]
    hosts = Host.query.filter(
        Host.id.notin_(hosts_in_cluster)
    ).all()
    if not hosts:
        return []
    return hosts


def ensure_swarm_cluster_init():
    if next((c for c in cluster_manager.clusters.values() if c.cluster_type == ClusterType.SWARM), None):
        return
    from mlsteam.core.host import get_master_host
    host = None
    try:
        host = get_master_host()
    except Exception:
        pass
    if not host or not host.authorize:
        return
    cluster_uuid = cluster_manager.gen_cluster_uuid()
    cluster_name = f'Cluster_{cluster_uuid}'
    cluster_manager.create_cluster(cluster_name, ClusterType.SWARM, cluster_uuid=cluster_uuid,
                                   node_list=[host], master_hostname=host.hostname)
