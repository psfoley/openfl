#!/bin/bash

if [ "$#" -ne 1 ];
then
    echo "Usage: create-collaborator FQDN"
    exit 1;
fi

FQDN=$1
subject_alt_name="DNS:$FQDN"

echo "Creating collaborator key pair with following settings: CN=$FQDN SAN=$subject_alt_name"

SAN=$subject_alt_name openssl req -new -config config/client.conf -out $FQDN.csr -keyout $FQDN.key -subj "/CN=$FQDN"
openssl ca -config config/signing-ca.conf -batch -in $FQDN.csr -out $FQDN.crt

mkdir -p $FQDN
mv $FQDN.crt $FQDN.key $FQDN
rm $FQDN.csr
