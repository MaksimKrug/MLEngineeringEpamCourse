FROM ubuntu:18.04
WORKDIR /workdir
COPY file.sh environment.yml ./
RUN chmod +x "./file.sh" && bash "./file.sh"
ENV PATH="/miniconda3/bin:$PATH"

# Create User
ARG USER_ID
ARG GROUP_ID
RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user
