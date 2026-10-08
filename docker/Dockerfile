# Release tag verified via Docker Hub API (amd64, arm64, ppc64le).
FROM condaforge/miniforge3:26.7.2-0

WORKDIR /lab

COPY environment.yml /lab/environment.yml
COPY platform /lab/platform
COPY student /lab/student
# The PyPI Linux torch wheel pulls several GB of CUDA libraries; the lab runs on CPU only.
RUN PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu mamba env create -f environment.yml \
    && mamba clean -afy

SHELL ["conda", "run", "-n", "day23_sensor_fusion", "/bin/bash", "-c"]

ENV DAY23_STUDENT_ROOT=/lab/student
ENV FUSION_LAB_PLATFORM=/lab/platform
ENV DAY23_HEADLESS=1
ENV DAY23_VIZ_SAVE_DIR=/lab/student/artifacts/viz

CMD ["conda", "run", "--no-capture-output", "-n", "day23_sensor_fusion", "fusion-run-lab", "--config", "student/config/paths.docker.yaml", "--fusion", "compare", "--max-frames", "5"]
