ARG PLATFORM=linux/amd64

FROM --platform=${PLATFORM} kartoza/postgis:13-3.1
COPY odc_ls8_sample.sql /docker-entrypoint-initdb.d/
