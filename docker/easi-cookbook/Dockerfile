ARG PLATFORM=linux/amd64
# TODO Replace with values suitable for your deployment environment.
# Note: You can use any Debian/Ubuntu based image you want.
FROM --platform=${PLATFORM} 444488357543.dkr.ecr.ap-southeast-2.amazonaws.com/easi-workflows-base:develop.latest

# This Dockerfile adds a non-root user with sudo access. Use the "remoteUser"
# property in devcontainer.json to use it. On Linux, the container user's GID/UIDs
# will be updated to match your local UID/GID (when using the dockerFile property).
# See https://aka.ms/vscode-remote/containers/non-root-user for details.
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
# Install coverage analysis, temp until enabled in base container
RUN /env/bin/pip install pytest-cov
# Create a non-root user to use if not already available - see https://aka.ms/vscode-remote/containers/non-root-user.
RUN if [ $(getent passwd $USERNAME) ]; then \
        # If exists, see if we need to tweak the GID/UID
        if [ "$USER_GID" != "1000" ] || [ "$USER_UID" != "1000" ]; then \
            groupmod --gid $USER_GID $USERNAME \
            && usermod --uid $USER_UID --gid $USER_GID $USERNAME \
            && chown -R $USER_UID:$USER_GID /home/$USERNAME; \
        fi; \
    else \
        # Otherwise ccreate the non-root user
        groupadd --gid $USER_GID $USERNAME \
        && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
        # Add sudo support for the non-root user
        && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME\
        && chmod 0440 /etc/sudoers.d/$USERNAME; \
    fi

ENTRYPOINT ["/bin/tini", "-s", "--", "with_bootstrap"]
CMD ["/bin/bash"]