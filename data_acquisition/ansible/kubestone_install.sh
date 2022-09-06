#!/bin/bash

KUBECONFIG=./fetched/config
KUBESTONE_PATCH_FILE=./tmp/kubestone-patch.yml


printf "\n##### Deploy Kubestone Operator...\n"
kubectl kustomize github.com/mcd01/kubestone-fork/config/default?ref=v0.5.1 | sed "s/xridge\/kubestone:latest/mcd01\/kubestone-fork:v0.5.1/" | kubectl create --kubeconfig ${KUBECONFIG} -f -

cat <<EOF > ${KUBESTONE_PATCH_FILE}
spec:
  template:
    spec:
      nodeSelector:
        mynodetype: "supporter"
EOF
kubectl patch deployment kubestone-controller-manager --patch-file ${KUBESTONE_PATCH_FILE} -n kubestone-system --kubeconfig ${KUBECONFIG}
