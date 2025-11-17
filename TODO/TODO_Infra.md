# TODO: Infrastructure / CI / Containerization

1. Docker
   - Create `Dockerfile` for the Flask app including Python dependencies and lightweight startup.
   - Create `docker-compose.yml` to wire app + Postgres + Redis (dev stack).

2. CI
   - Add `.github/workflows/ci.yml` to run tests and lint on push/PR.

3. Deployment manifests
   - Add Kubernetes manifests or Helm chart (optional) for production deployment.

4. Secrets management
   - Document environment variables and add sample `.env.example`.

5. Infra automation
   - Add Terraform/CloudFormation stubs if you plan to deploy cloud infra.
