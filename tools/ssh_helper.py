"""SSH helper — run commands or transfer files on the remote machine.

Credentials are read from environment variables (or a local .env file):
    SSH_HOST  — remote server IP / hostname
    SSH_USER  — remote username
    SSH_PASS  — remote password

Create a local .env file (git-ignored) with:
    SSH_HOST=<your-ip>
    SSH_USER=<your-user>
    SSH_PASS=<your-password>
"""
import os
import paramiko

def _load_env():
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    env_path = os.path.normpath(env_path)
    if os.path.exists(env_path):
        for line in open(env_path, encoding='utf-8'):
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip())

_load_env()

HOST    = os.environ.get("SSH_HOST", "")
USER    = os.environ.get("SSH_USER", "")
PASS    = os.environ.get("SSH_PASS", "")
PROJECT = "/tmp/glad_quant_test"
CONDA_RUN = (
    "source /datasdd/home/lizx/anaconda3/etc/profile.d/conda.sh && "
    "conda activate pytorch_gpu && "
    f"cd {PROJECT} && "
)


def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PASS, timeout=15)
    return c


def run(cmd, timeout=120):
    """Run a shell command; returns (stdout, stderr, exit_code)."""
    with connect() as c:
        _, stdout, stderr = c.exec_command(cmd, timeout=timeout, get_pty=False)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        code = stdout.channel.recv_exit_status()
    return out, err, code


def run_in_project(cmd, timeout=120):
    """Run cmd inside conda env at PROJECT dir."""
    full = CONDA_RUN + cmd
    return run(f'bash -lc "{full}"', timeout)


def put(local_path, remote_path):
    """Upload a single file."""
    with connect() as c:
        with c.open_sftp() as sftp:
            sftp.put(local_path, remote_path)


def get_file(remote_path):
    """Read a remote file as string."""
    with connect() as c:
        with c.open_sftp() as sftp:
            with sftp.open(remote_path, "r") as f:
                return f.read().decode("utf-8", errors="replace")


def put_text(text, remote_path):
    """Write text content to a remote file."""
    import io
    with connect() as c:
        with c.open_sftp() as sftp:
            sftp.putfo(io.BytesIO(text.encode("utf-8")), remote_path)


if __name__ == "__main__":
    # Quick connectivity test
    out, err, code = run("echo hello && hostname && nvidia-smi --query-gpu=name --format=csv,noheader")
    print(out, err)
