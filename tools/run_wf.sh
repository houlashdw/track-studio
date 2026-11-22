#!/bin/bash
# tools/run_wf.sh <cmd> [ksrc] — запустить modal.yml и дождаться
A=(-f ref=main -f "inputs[cmd]=$1"); [ -n "$2" ] && A+=(-f "inputs[ksrc]=$2"); [ -n "$3" ] && A+=(-f "inputs[tenv]=$3")
for t in 1 2 3; do gh api repos/houlashdw/track-studio/actions/workflows/modal.yml/dispatches "${A[@]}" && break; sleep 5; done
sleep 10; ID=$(gh api "repos/houlashdw/track-studio/actions/workflows/modal.yml/runs?per_page=1" --jq '.workflow_runs[0].id')
for i in $(seq 1 100); do r=$(gh api repos/houlashdw/track-studio/actions/runs/$ID --jq '[.status,.conclusion]|join(" ")'); case "$r" in completed*) break;; esac; sleep 10; done; echo "$1 $ID $r"
