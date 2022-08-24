FROM ubuntu:18.04
WORKDIR /workdir
COPY file.sh environment.yml jupyter_run.sh ./

COPY jupyter_notebook_config.json .

RUN chmod +x "./file.sh" && bash "./file.sh"
ENV PATH="/miniconda3/bin:$PATH"

# Create User
ARG USER_ID
ARG GROUP_ID
RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user

ENTRYPOINT bash jupyter_run.sh