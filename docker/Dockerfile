FROM python:3.12.12-slim@sha256:f3fa41d74a768c2fce8016b98c191ae8c1bacd8f1152870a3f9f87d350920b7c

# No target repository or provider credentials enter the image. This dependency
# set supports the bundled demo and the frozen Arrow development cases.
COPY requirements.lock /opt/requirements.lock
RUN pip install --no-cache-dir -r /opt/requirements.lock

USER 65534:65534
WORKDIR /tmp
