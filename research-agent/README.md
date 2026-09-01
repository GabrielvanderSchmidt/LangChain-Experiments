Run application container:

```
podman run -it --rm --name agent-env -v $PWD:/workspace:rw --network bridge agent-env:latest
```

Run Ollama backend:

```
podman volume create ollama_storage

podman run -d --name ollama -v ollama_storage:/root/.ollama -p 11434:11434 --network bridge docker.io/ollama/ollama

curl http://localhost:11434/api/pull -d '{
  "name": "qwen3:4b-q4_K_M"
}'
```

Connect containers' network:
```
podman network create agents-net
podman network connect agents-net agent-env
podman network connect agents-net ollama
```

