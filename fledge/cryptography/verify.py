# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.


import base64
import cryptography.exceptions
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.serialization import load_pem_public_key


def verify_file(path,
                sig_path,
                public_key='public.pem',
                ):
    """Verify the certificate file

    Args:
        sig_path: Filename of the signed certificate
        public_key: Filename for the public key

    Returns:
        bool: True if public key is validated by certificate

    """
    # Load the public key.
    with open(public_key, 'rb') as f:
        public_key = load_pem_public_key(f.read(), default_backend())

    # Load the payload contents and the signature.
    with open(path, 'rb') as f:
        contents = f.read()
    with open(sig_path, 'rb') as f:
        signature = base64.b64decode(f.read())

    # Perform the verification.
    try:
        public_key.verify(
            signature,
            contents,
            padding.PSS(
                mgf = padding.MGF1(hashes.SHA256()),
                salt_length = padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        return True
    except cryptography.exceptions.InvalidSignature as e:
        print('ERROR: Payload and/or signature files failed verification!')
        return False
