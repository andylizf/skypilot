"""Trace instance provisioning."""
import os
import json
from typing import Any, Dict, List, Optional, Tuple

from sky import sky_logging
from sky.provision import common
from sky.utils import status_lib

TRACE_CLOUD_RECORD_FILE = '@temp/trace.json'
CURRENT_TICK_KEY = 'current-tick'
if not os.path.exists(TRACE_CLOUD_RECORD_FILE):
    with open(TRACE_CLOUD_RECORD_FILE, 'w') as f:
        json.dump({CURRENT_TICK_KEY: 0}, f)

POLL_INTERVAL = 5
QUERY_PORTS_TIMEOUT_SECONDS = 30

logger = sky_logging.init_logger(__name__)


def run_instances(region: str, cluster_name_on_cloud: str,
                  config: common.ProvisionConfig) -> common.ProvisionRecord:
    """Runs instances for the given cluster."""
    with open(TRACE_CLOUD_RECORD_FILE, 'r') as f:
        d = json.load(f)
    if d[CURRENT_TICK_KEY] >= 10:
        raise ValueError('NoCapacityError')
    if cluster_name_on_cloud not in d:
        d[cluster_name_on_cloud] = {
            'head': f'{cluster_name_on_cloud}-head',
            'workers': [f'{cluster_name_on_cloud}-worker-{i}' for i in range(config.count - 1)]
        }
    with open(TRACE_CLOUD_RECORD_FILE, 'w') as f:
        json.dump(d, f)
    return common.ProvisionRecord(
        provider_name='trace',
        cluster_name=cluster_name_on_cloud,
        region=region,
        zone=config.provider_config['availability_zone'],
        head_instance_id=d[cluster_name_on_cloud]['head'],
        resumed_instance_ids=[],
        created_instance_ids=d[cluster_name_on_cloud]['workers'])


def wait_instances(region: str, cluster_name_on_cloud: str,
                   state: Optional[status_lib.ClusterStatus]) -> None:
    del region, cluster_name_on_cloud, state


def stop_instances(
    cluster_name_on_cloud: str,
    provider_config: Optional[Dict[str, Any]] = None,
    worker_only: bool = False,
) -> None:
    raise NotImplementedError()


def terminate_instances(
    cluster_name_on_cloud: str,
    provider_config: Optional[Dict[str, Any]] = None,
    worker_only: bool = False,
) -> None:
    """See sky/provision/__init__.py"""
    del provider_config, worker_only  # unused
    with open(TRACE_CLOUD_RECORD_FILE, 'r') as f:
        d = json.load(f)
    d.pop(cluster_name_on_cloud, None)
    with open(TRACE_CLOUD_RECORD_FILE, 'w') as f:
        json.dump(d, f)


def get_cluster_info(
        region: str,
        cluster_name_on_cloud: str,
        provider_config: Optional[Dict[str, Any]] = None) -> common.ClusterInfo:
    del region  # unused
    with open(TRACE_CLOUD_RECORD_FILE, 'r') as f:
        d = json.load(f)
    running_instances = d.get(cluster_name_on_cloud, {})
    instances: Dict[str, List[common.InstanceInfo]] = {}
    head_instance_id = running_instances.get('head', None)
    instance_ids = running_instances.get('workers', []) + [head_instance_id]
    for instance_id in instance_ids:
        instances[instance_id] = [
            common.InstanceInfo(
                instance_id=instance_id,
                internal_ip=f'ip-{instance_id}',
                external_ip=f'ip-{instance_id}',
                ssh_port=22,
                tags={},
            )
        ]

    return common.ClusterInfo(
        instances=instances,
        head_instance_id=head_instance_id,
        provider_name='trace',
        provider_config=provider_config,
    )


def query_instances(
    cluster_name: str,
    cluster_name_on_cloud: str,
    provider_config: Optional[Dict[str, Any]] = None,
    non_terminated_only: bool = True,
) -> Dict[str, Tuple[Optional['status_lib.ClusterStatus'], Optional[str]]]:
    """See sky/provision/__init__.py"""
    del cluster_name  # unused
    assert provider_config is not None, (cluster_name_on_cloud, provider_config)
    with open(TRACE_CLOUD_RECORD_FILE, 'r') as f:
        d = json.load(f)
    running_instances = d.get(cluster_name_on_cloud, {})
    head_instance_id = running_instances.get('head', None)
    instance_ids = running_instances.get('workers', []) + [head_instance_id]

    statuses: Dict[str, Tuple[Optional['status_lib.ClusterStatus'],
                              Optional[str]]] = {}
    for inst_id in instance_ids:
        statuses[inst_id] = (status_lib.ClusterStatus.UP, None)
    return statuses


def cleanup_ports(
    cluster_name_on_cloud: str,
    ports: List[str],
    provider_config: Optional[Dict[str, Any]] = None,
) -> None:
    del cluster_name_on_cloud, ports, provider_config  # Unused.

