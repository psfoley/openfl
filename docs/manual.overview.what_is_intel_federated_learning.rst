What is Intel® Federated Learning?
==================================

By leveraging the security provided by Intel:superscript:`®` SGX and the ease of deployment
provided by Graphene, Federated Learning can be protected from adversarial
attacks that are well documented in the literature. With Intel:superscript:`®` SGX on
every node in the federation, risks are mitigated even if the nodes are
not fully-controlled by the federation owner.

.. image:: images/diagram_fl.png

Previous attacks have shown that adversaries may be able to steal the model,
reconstruct data based on the model updates, and/or prevent convergence of
the training when using untrusted nodes
(`Bagdasaryan, Veit, Hua, Estrin, & Shmatikov, 2018<https://arxiv.org/abs/1807.00459>`;
`Bhagoji, Chakraborty, Supriyo, & Calo, 2018<https://arxiv.org/abs/1811.12470>`).
With Intel® Federated Learning protected via Intel® SGX,
adversaries are unable to use the model and unable to adapt their
attacks because the actual training is only visible to those with an
approved key.

Additionally, Intel:superscript:`®` SGX allows developers to require attestation
from collaborators which proves that the collaborator actually
ran the expected code within the enclave. Attestation can either 
be done via a trusted Intel server or by the developer’s own server.
This stops attackers from injecting their own code into the federated training.

.. image:: images/why-sgx-for_fl.png
