---
title: Demo of gRPC with mutual TLS 
author: Weilin Xu <weilin.xu@intel.com>
date: 09/04/2019
---

# Demo of gRPC with mutual TLS 
We demonstrate how to generate certificates and establish a mutual TLS connection with gRPC.

## Generate TLS Certificates
Let's generate the certificates.
We would use the tool [**certstrp**](https://github.com/square/certstrap) from Square instead of `openssl` to simplify the procedure and avoid dangerous configurations.

First, we establish a Certificate Authority (CA) by generating the necessary files. The common name of a CA must be a domain name, not IP address.

```console
$ certstrap init --common-name "localhost"
Enter passphrase (empty for no passphrase):

Enter same passphrase again:

Created out/localhost.key
Created out/localhost.crt
Created out/localhost.crl
```

The command produces three files: the top-secret private key (localhost.key), the certificate (localhost.crt) that you will share with the world, and the Certificate Revocation List (localhost.crl) which you won't need for now.

Next, each individual will generate their key pairs associated with an IP address or a domain name.
```console
$ certstrap request-cert --common-name "spr-gpu02"
Enter passphrase (empty for no passphrase):

Enter same passphrase again:

Created out/spr-gpu02.key
Created out/spr-gpu02.csr
```

The commond produces two files: the top-secret private key (.key) that only an individual itself knows, and the Certificate Sign Request (.csr) that it will share with the CA later to get an individual certificate.

We generate the other key pair associated with an IP address below:
```console
$ certstrap request-cert -ip 10.24.14.200
Enter passphrase (empty for no passphrase):

Enter same passphrase again:

Created out/10.24.14.200.key
Created out/10.24.14.200.csr
```

Finally, the CA signs individual certificates with it's CA private key given the individual Certificate Sign Request (.csr) files.

```console
$ certstrap sign spr-gpu02 --CA localhost
Created out/spr-gpu02.crt from out/spr-gpu02.csr signed by out/localhost.key

$ certstrap sign 10.24.14.200 --CA localhost
Created out/10.24.14.200.crt from out/10.24.14.200.csr signed by out/localhost.key
```

Each party (either TLS server or client) would need to have three files to establish the mutual TLS: 
* The CA certificate (localhost.crt) as the root of trust.
* The individual certificate (spr-gpu02.crt) for others to authenticate and encrypt the traffic.
* The individual private key (spr-gpu02.key) to decrypt the traffic.


> Also, the client side at Intel would need to unset all the proxy env for debugging. 
> ```shell
> unset http_proxy
> unset https_proxy
> ```


## Mutual TLS Helloworld

### Compile the helloworld protobuf

```shell
python -m grpc_tools.protoc  --python_out=./protos --grpc_python_out=./protos ./protos/helloworld.proto -I ./protos/
```

### Run the server code on spr-gpu02
```console
$ python server_remote.py
```

### Run the client code on the other machine
```console
$ python client_remote.py
Greeter client received: Hello, you!
```