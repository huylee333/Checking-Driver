"""T03. The real image, as a container (Automation, System).

Checks what the published image promises: it starts, answers, reports healthy, runs as a normal user and keeps data in
a volume across a restart. Run it with CD_IMAGE=<image>.
"""
from docker_helpers import docker, http_get, wait_until


def test_the_container_starts_and_answers_without_weights(start_container):
    c = start_container()
    c.wait_web()
    status = c.status()
    assert status['ready'] is False and status['video'] is False and status['fusion'] is False
    assert http_get(c.url + '/api/job/none')[0] == 404


def test_docker_reports_the_container_healthy(start_container):
    c = start_container()
    c.wait_web()
    # the HEALTHCHECK runs every 30 s, so the first verdict arrives within about a minute
    wait_until(lambda: c.inspect('{{.State.Health.Status}}') == 'healthy', timeout=150, interval=5,
               what='HEALTHCHECK healthy')


def test_the_process_runs_as_a_normal_user_not_root(start_container):
    c = start_container()
    c.wait_web()
    assert c.inspect('{{.Config.User}}') == 'user'
    assert c.exec('id', '-u') == '1000'
    assert c.exec('sh', '-c', 'touch /should-not-be-writable 2>/dev/null; echo $?') != '0'     # cannot write to /


def test_the_container_listens_on_7860_whatever_host_port_is_published(start_container):
    c = start_container()
    c.wait_web()
    assert c.exec('python', '-c', "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:7860/').status)") == '200'


def test_files_in_the_weights_volume_survive_a_restart(start_container, volume):
    vol = volume()
    c = start_container('-v', f'{vol}:/home/user/app/weights')
    c.wait_web()
    c.exec('sh', '-c', 'echo kept > /home/user/app/weights/marker.txt')          # the volume is writable by the user
    docker('restart', c.name)
    c.wait_web()
    assert c.exec('cat', '/home/user/app/weights/marker.txt') == 'kept'


def test_no_secret_is_baked_into_the_image_environment(start_container):
    c = start_container()
    env = c.inspect('{{json .Config.Env}}')
    for word in ('HF_TOKEN', 'DOCKER', 'PASSWORD', 'SECRET', 'hf_'):
        assert word not in env
    history = docker('history', '--no-trunc', '--format', '{{.CreatedBy}}', docker('inspect', '--format', '{{.Image}}', c.name))
    assert 'hf_' not in history and 'dckr_pat' not in history
