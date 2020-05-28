#!/bin/bash

if [ "$#" -ne 2 ];
then
    echo "Usage: create-aggregator FQDN IP_ADDRESS"
    exit 1;
fi

FQDN=$1
IP_ADDRESS=$2
subject_alt_name="DNS:$FQDN,IP:$IP_ADDRESS"

echo "Creating debug client key pair with following settings: CN=$FQDN SAN=$subject_alt_name"

SAN=$subject_alt_name openssl req -new -config config/server.conf -subj "/CN=$FQDN" -out $FQDN.csr -keyout $FQDN.key
openssl ca -config config/signing-ca.conf -batch -extensions server_ext -in $FQDN.csr -out $FQDN.crt

mkdir -p $FQDN
mv $FQDN.crt $FQDN.key $FQDN
rm $FQDN.csr
