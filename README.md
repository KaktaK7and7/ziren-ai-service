# ziren-ai-service

Backend для ассистента:
- persona
- user memory
- chat
- OpenAI API
- PostgreSQL

## Run locally

```bash
cp .env.example .env
pip install -r requirements.txt
uvicorn app.main:app --reload
```

`AI_INTERNAL_TOKEN` is required. Every endpoint except `/health` accepts calls
only when the same secret is sent in the `X-Ziren-Internal-Token` header. The
auth gateway and AI service must use the same independently generated secret.
