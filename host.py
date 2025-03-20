import os
import re
import json

from sqlalchemy.orm.attributes import flag_modified
from sqlalchemy.sql.expression import or_

from mlsteam.core import ha
from mlsteam.core.const import AGENT_INSTALLER_PATH, DATA_ROOT
from mlsteam.core.const import DEFAULT_CLUSTER_NAMESPACE
from mlsteam.core.consumption.host import add_hostdown
from mlsteam.core.consumption.host import add_hostup
from mlsteam.core.models import Host
from mlsteam.core.resource_tag import resource_tag_list
from mlsteam.core.utils import auditlog
from mlsteam.core.utils.container import get_joined_network_id, get_real_path
from mlsteam.core.utils.files import hosts_record
from mlsteam.exception.base import HostException
from mlsteam.exception.host import HostConnectError, HostNotFound
from mlsteam.exception.feature import FeatureDisabled, FeatureExceed
from mlsteam.log import logger
from mlsteam.utils.license import get_hostid
from mlsteam.utils.feature_flag import feature_config_get
from mlsteam.webapp import db

NETDATA_TEMPLATE = "netdata_template"


def host_update(host_id, tags=None):
    """Updates resource tags for a host

    `tags` could be in the following formats:
    - None (no change)
    - a string with tag Ids separated by `;`, `,`, or `|`
    - a list of tag Ids

    To clear all resource tags for a host, assign `tags` to an empty string or list.
    """
    from mlsteam.resource import worker_resources
    h = host_get_by_id(host_id)
    try:
        changed = False
        if tags is not None:
            resource_tags = resource_tag_list()

            if isinstance(tags, str):
                try:
                    tags = [int(x) for x in re.split('[;,|]', tags.strip()) if x]
                except ValueError as e:
                    raise ValueError(f'invalid resource tag Ids {tags}') from e
            elif isinstance(tags, list):
                tags = [int(x) for x in tags if x]

            for tag_id in tags:
                if str(tag_id) not in resource_tags:
                    raise ValueError(f'resource tag Id {tag_id} not found')
            h.tags = tags
            flag_modified(h, 'tags')
            changed = True

        if changed:
            db.session.commit()
            w = worker_resources.get_worker(h.hostname)
            if w:
                if tags is not None:
                    w.tags = tags
        return h
    except Exception as e:
        db.session.rollback()
        raise e


# create in DB, called by socketio agent registered
def host_create(host_id, hostname, resources, sid='', version='', authorize=False, skip_host_record=False):
    ip = resources.get('ip')
    if not ip or not resources:
        return
    h = Host.query.filter(Host.host_id == host_id).first()
    if h:
        h.resources = resources
        h.ip = ip
        h.hostname = hostname
        h.sid = sid
        h.version = version
        db.session.commit()
        if not skip_host_record:
            hosts_record(ip, hostname)
        return h.authorize, h.in_cluster
    logger.debug("create host ({}, {}, {}, {})".format(host_id, hostname, ip, resources))
    h = Host(host_id=host_id, hostname=hostname, ip=ip,
             resources=resources, authorize=authorize, sid=sid, version=version)
    db.session.add(h)
    db.session.commit()
    if not skip_host_record:
        hosts_record(ip, hostname)
    return False, False


def host_authorize_update(host_id, authorize=True):
    from mlsteam.core.cluster import cluster_manager
    try:
        h = Host.query.filter(Host.host_id == host_id).first()
        if not h:
            raise HostNotFound(f"Host {host_id} not found", host=host_id)
        if h.in_cluster and not cluster_manager.get_cluster(h.cluster_hosts.cluster_id).has_agent:
            raise ValueError('Cannot authorize a host without agent')

        infras_limit = feature_config_get('infras_limit', 0)
        if infras_limit == -1:
            pass
        elif infras_limit == 0:
            raise FeatureDisabled('Feature infrastructure is disabled', 'infra limit')
        else:
            all_auth_nodes = Host.query.filter(Host.authorize is True).all()
            if len(all_auth_nodes) >= infras_limit:
                raise FeatureExceed('Feature infrastructure limit exceeded', 'infra limit')

        if authorize and not h.is_master:
            if len(cluster_manager.clusters) == 0:
                raise ValueError('require at least one cluster')

        if not authorize:
            host_disconnect(host_id)

        if not h.in_cluster and authorize:
            if len(cluster_manager.clusters):
                try:
                    list(cluster_manager.clusters.values())[0].node_join(h, authorize)
                except Exception:
                    import traceback
                    logger.error(traceback.format_exc())

        # to DB
        if h.authorize != authorize:
            h.authorize = authorize
            db.session.commit()
    except Exception as e:
        raise HostException(f"Authorized host {host_id} failed {str(e)}") from e


def host_get_by_ip(ip):
    host = Host.query.filter(Host.ip == ip).first()
    if not host:
        raise HostNotFound(f"Host {ip} not found", host=ip)
    return host


def host_get_by_id(hid):
    host = (Host.query.filter(Host.id == hid).first() or Host.query.filter(Host.host_id == hid).first())
    if not host:
        raise HostNotFound(f"Host {hid} not found", host=hid)
    return host


def host_get_ip(hostname):
    host = Host.query.filter(Host.hostname == hostname).first()
    if not host:
        return ''
    return host.ip


def host_list() -> list[Host]:
    # list from db
    hosts = Host.query.filter().all()
    if not hosts:
        return []
    return hosts


# deleted by admin
def host_delete(host_id, db_delete=False):
    from mlsteam.core.cluster import cluster_manager
    from mlsteam.core.utils.socket import mlsteam_socket
    host = host_get_by_id(host_id)
    hostname = host.hostname
    has_agent = True
    if host.in_cluster:
        cluster = cluster_manager.get_cluster(host.cluster_hosts.cluster_id)
        has_agent = cluster.has_agent
        cluster.node_leave(host.cluster_hosts.id)
    if has_agent:
        mlsteam_socket.disconnect(sid=host.sid)
    if db_delete:
        db.session.delete(host)
        db.session.commit()
    return hostname


def host_disconnect(host_id):
    from mlsteam.core.cluster import cluster_manager
    from mlsteam.core.utils.socket import mlsteam_socket
    host = host_get_by_id(host_id)
    try:
        if host.in_cluster and not cluster_manager.get_cluster(host.cluster_hosts.cluster_id).has_agent:
            raise ValueError('no agent for this host')
        mlsteam_socket.disconnect(sid=host.sid)
    except Exception as e:
        logger.warning('host_disconnect failed, {}'.format(e))


def prepare_host_installer(host):
    from mlsteam.core.certificate import certificate_list

    # update hostid
    hostid = get_hostid()
    data_dir = get_real_path(DATA_ROOT)

    # print(host)  # 127.0.0.1 or 127.0.0.1:8080

    if ':' in host:
        ip, port = host.split(":", 1)
    else:
        ip, port = host, 80

    https = 'false'
    for cert in certificate_list():
        if re.match(cert.domain + '/*$', host):
            https = 'true'
            port = 443
            break
    if not os.path.exists(AGENT_INSTALLER_PATH):
        raise OSError("Installer script not found")

    os.system("sed -i 's|^HOST_ID=.*|HOST_ID=\"{}\"|g' {}".format(hostid, AGENT_INSTALLER_PATH))  # nosec
    os.system("sed -i 's|^DATA_DIR=.*|DATA_DIR=\"{}\"|g' {}".format(data_dir, AGENT_INSTALLER_PATH))  # nosec
    os.system("sed -i 's|^HTTPS=.*|HTTPS=\"{}\"|g' {}".format(https, AGENT_INSTALLER_PATH))  # nosec
    os.system("sed -i 's|^IP=.*|IP=\"{}\"|g' {}".format(ip, AGENT_INSTALLER_PATH))  # nosec
    os.system("sed -i 's|^PORT=.*|PORT=\"{}\"|g' {}".format(port, AGENT_INSTALLER_PATH))  # nosec
    os.system("sed -i 's|^NAMESPACE=.*|NAMESPACE=\"{}\"|g' {}".format(  # nosec
        DEFAULT_CLUSTER_NAMESPACE, AGENT_INSTALLER_PATH))


def host_upgrade(host_id):
    import base64

    from mlsteam.core.cluster import cluster_manager
    from mlsteam.core.utils.socket import mlsteam_socket
    try:
        with open(AGENT_INSTALLER_PATH, 'rb') as f:
            bytes_file = base64.b64encode(f.read())
            host = host_get_by_id(host_id)
            if not host.alive:
                raise HostConnectError(f"Agent on host {host.hostname} is disconnected", host=host_id)
            if host.in_cluster and not cluster_manager.get_cluster(host.cluster_hosts.cluster_id).has_agent:
                raise ValueError('Cannot upgrade a host without agent')
            mlsteam_socket.node_upgrade({'encoded_file': bytes_file}, host.hostname)
            return host.hostname
    except Exception as e:
        raise e


def host_version(host_id):
    try:
        host = host_get_by_id(host_id)
        if not host.alive:
            raise HostConnectError(f"Agent on host {host.hostname} is disconnected", host=host_id)
        return host.version
    except Exception as e:
        logger.error('get host version failed : {}'.format(e))
        raise e


# agent disconnected
def host_offline(sid_or_hostid):
    from mlsteam.core.cluster import cluster_manager
    from mlsteam.resource import worker_resources
    host = Host.query.filter(
        or_(Host.sid == sid_or_hostid, Host.host_id == sid_or_hostid)
    ).first()
    if not host:
        return

    # set not alive only
    worker_resources.on_node_leave(host)
    hostname = host.hostname
    has_agent = True
    if host.in_cluster:
        cluster = cluster_manager.get_cluster(host.cluster_hosts.cluster_id)
        has_agent = cluster.has_agent
    if has_agent:
        if host.is_authorized:
            auditlog.info('SYSTEM', auditlog.SYSTEM, 'Agent {} disconnected', hostname)
            # add hostdown log
            add_hostdown(hostname)
    else:
        # add hostdown log
        add_hostdown(hostname)


# back online from offline
# agent connected
def host_online(host_id, **kwargs):
    from mlsteam.core.cluster import ClusterType, cluster_manager
    from mlsteam.core.utils.socket import mlsteam_socket
    from mlsteam.resource import worker_resources
    from mlsteam.initapp import app
    host = host_get_by_id(host_id)

    # set alive
    w = worker_resources.on_node_join(host, **kwargs)
    # add hostup log
    spec = w.spec()
    add_hostup(
        hostname=w.hostname,
        cpu=spec.get('cpu', -1),
        memory=spec.get('memory', -1),
        gpus=spec.get('gpus', {}),
        suppress_log=kwargs.get('suppress_log', False)
    )

    # in cluster
    has_agent = True
    if host.in_cluster:
        cluster = cluster_manager.get_cluster(host.cluster_hosts.cluster_id)
        has_agent = cluster.has_agent
        if cluster and cluster.cluster_type == ClusterType.SWARM:
            cluster.setup_overlay_network()
            cluster.node_setup(host)

    # single node
    if has_agent:
        if host.is_master:
            default_netid = get_joined_network_id(app.config['MLSTEAM_NETWORK'])
            if default_netid:
                mlsteam_socket.join_network(network_name=app.config['MLSTEAM_NETWORK'],
                                            network_id=default_netid, host=host.hostname)
        mlsteam_socket.components_check(host.hostname)
        auditlog.info('SYSTEM', auditlog.SYSTEM, 'Agent {} connected', host.hostname)


def host_component_status(sid, components):
    '''see Agent: check_component_ready()'''
    from mlsteam.core import nginx
    from mlsteam.resource import worker_resources
    h = Host.query.filter(Host.sid == sid).first()
    if not h:
        return
    w = worker_resources.get_worker(h.hostname)
    if not w:
        return
    ready = w.ready
    for component in components:
        if 'name' not in component or 'status' not in component:
            logger.error('[on_components]: invalid component {}'.format(component))
            continue
        if component['name'] not in ['netdata', 'swarm', 'term_service', 'rocm_smi_monitor']:
            logger.error('[host_component_status]: unknown component: {}'.format(component))
            continue
        w.components[component['name']] = component['status']
        if component['name'] == 'netdata' and component['status'] == 'running':
            # update nginx
            nginx.conf_create(NETDATA_TEMPLATE, "NETDATA_{}".format(h.hostname), {"hostname": h.hostname})
    if not ready and w.ready:
        logger.info('Agent {} is ready to receive task, {}'.format(h.hostname, w.components))


def host_components_error(sid, error):
    from mlsteam.resource import worker_resources
    h = Host.query.filter(Host.sid == sid).first()
    if not h:
        return
    w = worker_resources.get_worker(h.hostname)
    if not w:
        return
    w._is_error = True
    logger.info("Agent {} components error: {}".format(
        h.hostname, dict(filter(lambda i: i[1] == 'error', w.components.items()))))


def host_reset(host_id):
    from mlsteam.core.utils.socket import mlsteam_socket
    try:
        host = host_get_by_id(host_id)
        if not host.alive:
            raise HostConnectError(f"Agent on host {host.hostname} is disconnected", host=host_id)
        mlsteam_socket.node_reset(host.hostname)
        return True
    except Exception as e:
        logger.error('host {} reset failed : {}'.format(host_id, e))


def get_master_ip():
    if ha.enabled():
        return ha.cluster_ip()
    else:
        return get_master_host().ip


def get_master_ip_none():
    try:
        return get_master_ip()
    except Exception:
        return None


def get_master_host():
    master = None
    for h in host_list():
        if h.is_master:
            master = h
    if not master:
        raise HostNotFound("Master not found", host='Master')
    return master


def host_gpu_disable(hid: str, gpu_id: str, disable: bool):
    from mlsteam.resource import worker_resources
    host = host_get_by_id(hid)
    worker = worker_resources.get_worker(host.hostname)
    if not worker:
        raise HostNotFound(f"Host {hid} not found", host=hid)
    worker_resources.check_preserve(override_gpu_disable={gpu_id: disable})
    logger.info('Set Host={} GPU={} Disable={}'.format(host.hostname, gpu_id, disable))
    worker.worker_gpus.update(gpu_id, disable=disable)
    # Save gpu blacklist into Host DB
    blacklist = json.loads(host.resources_blacklist) if host.resources_blacklist else {}
    gpu_blist = blacklist.get('gpu', [])
    if disable:
        if gpu_id not in gpu_blist:
            gpu_blist.append(gpu_id)
    else:
        if gpu_id in gpu_blist:
            gpu_blist.remove(gpu_id)
    host.resources_blacklist = json.dumps({'gpu': gpu_blist})
    db.session.add(host)
    db.session.commit()


def host_gpu_slice(hid: str, gpu_uuid: str, slice_policy: int):
    from mlsteam.core.utils.socket import send_actions
    from mlsteam.resource import worker_resources

    host = host_get_by_id(hid)
    if not host.alive:
        raise HostConnectError(f"Agent on host {host.hostname} is disconnected", host=hid)
    if worker_resources.is_gpu_any_inuse(host):
        raise ValueError("host gpus are occupied, please stop tasks first")
    # if worker_resources.is_gpu_inuse(host, gpu_uuid):
    #     raise ValueError("selected gpu is use, please stop tasks first")
    partition = {
        0: {},
        1: {i: dict(memory=0.500, capacity=0.500) for i in range(2)},
        2: {i: dict(memory=0.333, capacity=0.333) for i in range(3)},
        3: {i: dict(memory=0.250, capacity=0.250) for i in range(4)},
        4: {i: dict(memory=0.200, capacity=0.200) for i in range(5)},
        5: {i: dict(memory=0.166, capacity=0.166) for i in range(6)},
        6: {i: dict(memory=0.142, capacity=0.142) for i in range(7)},
        7: {i: dict(memory=0.125, capacity=0.125) for i in range(8)},
        8: {i: dict(memory=0.111, capacity=0.111) for i in range(9)},
        9: {i: dict(memory=0.100, capacity=0.100) for i in range(10)},
    }
    if slice_policy not in partition:
        raise ValueError('invalid slice_policy {}'.format(slice_policy))
    slice_partition = partition[slice_policy]
    nvidia_mig_action = [dict(
        name="GPU {} set slice".format(gpu_uuid),
        nvidia_slice=dict(
            state="present" if slice_partition else "absent",
            gpu_uuid=gpu_uuid,
            slice_partition=slice_partition,
        )
    )]
    logger.info('Set Host={} GPU={} Slices={}({})'.format(host.hostname, gpu_uuid, slice_policy, slice_partition))
    res = send_actions(host.hostname, "set gpu slice", nvidia_mig_action, deffered=300)
    if res.get('rc'):
        raise SystemError(f"update host gpu slice failed: {res.get('msg')}")

    resource_action = [dict(
        name="Get agent resources",
        resources=dict()
    )]
    res = send_actions(host.hostname, "get resources", resource_action, deffered=300)
    if res.get('rc'):
        raise SystemError(f"get host resources failed: {res.get('msg')}")

    resources = res.get('msg')[0]
    # Update worker gpu resources
    worker_resources.update_gpu_resource_spec(host, resources)


def host_gpu_mig(hid: str, gpu_uuid: str, mig_policy: int):
    from mlsteam.resource import worker_resources

    host = host_get_by_id(hid)
    if not host.alive:
        raise HostConnectError(f"Agent on host {host.hostname} is disconnected", host=hid)
    if worker_resources.is_gpu_any_inuse(host):
        raise ValueError("host gpus are occupied, please stop tasks first")

    worker_resources.check_preserve(override_gpu_mig={gpu_uuid: mig_policy})
    if not host.resources.get('nvidia_mig_capable') == 'true':
        raise RuntimeError('not support mig')
    memory = host.resources.get('nvidia.com', {}).get('nvidia.com/gpu.memory')
    if not memory:
        raise RuntimeError('cannot get gpu memory size')
    gb = int(memory) // (1024 * 8)
    partition = {
        0: "all-disabled",  # Disable MIG
        1: "all-3g.71gb",  # [3g.20gb] x 2
        2: "all-2g.35gb",  # [2g.10gb] x 4
        3: "all-balanced",  # [3g.20gb] x 1 [2g.10gb] x 2 [1g.5gb] x 2
        6: "all-1g.18gb",  # [1g.5gb] x 7
    }
    if mig_policy not in partition:
        raise ValueError('invalid mig_policy {}'.format(mig_policy))
    from mlsteam.core.cluster import cluster_manager
    cluster = cluster_manager.get_default_k8s_cluster()  # also check cluster existence
    mig_partition = partition[mig_policy]
    cluster.k8s_tool.create_node_label(host.hostname, 'nvidia.com/mig.config', mig_partition)
    logger.info('Set Host={} MIG={}({})'.format(host.hostname, mig_policy, mig_partition))


def host_info_extend(host_info: dict, resources=None):
    from mlsteam.core.tasks.helper import TaskOperator
    host = host_info.copy()

    # resources
    if resources is not None:
        compute_allocated = compute_total = 0
        memory_allocated = memory_total = 0
        gpu_allocated = gpu_total = 0
        tasks = []
        if host['name'] in resources:
            host_resources = resources[host['name']]
            compute_allocated = host_resources["cpu"]["total"] - host_resources["cpu"]["remaining"]
            compute_total = host_resources["cpu"]["total"]
            memory_allocated = host_resources["memory"]["total"] - host_resources["memory"]["remaining"]
            memory_total = host_resources["memory"]["total"]
            gpu_allocated = host_resources["gpu"]["total"]["any"] - host_resources["gpu"]["remaining"]["any"]
            gpu_total = host_resources["gpu"]["total"]["any"]
            for holder in host_resources["holders"]:
                # ex. { "uuid": "u6203ede", ... }
                ruuid = holder['uuid']
                run = TaskOperator.task_get(ruuid)
                if run:
                    tasks.append(run.serialize_simple)
        host['compute_allocated'] = compute_allocated
        host['compute_total'] = compute_total
        host['memory_allocated'] = memory_allocated
        host['memory_total'] = memory_total
        host['gpu_allocated'] = gpu_allocated
        host['gpu_total'] = gpu_total
        host['tasks'] = tasks

    # tags

    # tags should be a list (dict-typed tags have been converted in data migration)
    resource_tags = resource_tag_list()
    host['tags'] = [resource_tags[str(tid)] for tid in (host['tags'] or []) if str(tid) in resource_tags]
    host['system_tags'] = []
    if ha.enabled():
        status = ha.status()
        if host['host_id'] == status['primary']:
            host['system_tags'].append('ha:active')
            host['system_tags'].append('vip:{}'.format(status['cluster_ip']))
        elif host['host_id'] == status['secondary']:
            host['system_tags'].append('ha:standby')
    else:
        if host['is_master']:
            host['system_tags'].append('role:master')
    if host['cluster_name']:
        host['system_tags'].append('cname:{}'.format(host['cluster_name']))
    if host['cluster_type']:
        host['system_tags'].append('ctype:{}'.format(host['cluster_type']))

    return host
