# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.


import base64
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa


def gen_keys(public_path='public.pem',
             private_path='private.key'):
    # Generate the public/private key pair.
    private_key = rsa.generate_private_key(
        public_exponent = 65537,
        key_size = 4096,
        backend = default_backend(),
    )

    # Save the private key to a file.
    with open('private.key', 'wb') as f:
        f.write(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

    # Save the public key to a file.
    with open('public.pem', 'wb') as f:
        f.write(
            private_key.public_key().public_bytes(
                encoding = serialization.Encoding.PEM,
                format = serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )


def sign_file(path,
              sig_path,
              private_key='private.key'):

    # Load the private key. 
    with open(private_key, 'rb') as key_file: 
        private_key = serialization.load_pem_private_key(
            key_file.read(),
            password = None,
            backend = default_backend(),
        )

    # Load the contents of the file to be signed.
    with open(path, 'rb') as f:
        payload = f.read()

    # Sign the payload file.
    signature = base64.b64encode(
        private_key.sign(
            payload,
            padding.PSS(
                mgf = padding.MGF1(hashes.SHA256()),
                salt_length = padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
    )
    with open(sig_path, 'wb') as f:
        f.write(signature)
