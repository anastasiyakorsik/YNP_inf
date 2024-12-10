FROM ynp_inf_base_new

WORKDIR /
COPY . /

WORKDIR /src

ENTRYPOINT ["python", "main.py"]
