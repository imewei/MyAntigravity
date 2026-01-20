---
name: api-design-principles
description: REST, GraphQL, and gRPC design standards for scalable interfaces.
version: 2.2.1
agents:
  primary: backend-architect
skills:
- rest-design
- graphql-schema
- versioning-strategy
- api-security
allowed-tools: [Read, Write, Task, Bash]
triggers:
- keyword:api
- keyword:rest
- keyword:openapi
---

# API Design Principles

// turbo-all

# API Design Principles

Architecting intuitive, consistent, and lasting interfaces.



## Strategy & Paradigms (Parallel)

// parallel

### Paradigm Selection

| Style | Use Case | Strength |
|-------|----------|----------|
| **REST** | Public API, CRUD | Cacheable, Simple, Standard. |
| **GraphQL** | Mobile, Complex Data | No Over/Under-fetching, Type System. |
| **gRPC** | Internal Microservices | Fast (Protobuf), Streaming, Typed. |

### REST Maturity

1.  **Level 1**: Resources (URI).
2.  **Level 2**: Verbs (GET/POST) + Codes (200/404).
3.  **Level 3**: HATEOAS (Hypermedia links).

// end-parallel

---

## Decision Framework

### Action Mapping

-   **List**: `GET /resources` (Paginated).
-   **Read**: `GET /resources/{id}`.
-   **Create**: `POST /resources` (Return 201 + Location header).
-   **Update**: `PATCH /resources/{id}` (Partial).
-   **Replace**: `PUT /resources/{id}` (Full idempotency).
-   **Delete**: `DELETE /resources/{id}` (Return 204).

---

## Core Knowledge (Parallel)

// parallel

### Constitutional AI Principles

1.  **Consistency (Target: 100%)**: Standardize Error formats and Case (snake_case vs camelCase).
2.  **Versioning (Target: 100%)**: Breaking changes must be a new version (URL or Header).
3.  **Security (Target: 100%)**: Rate limits and Auth on *every* endpoint.

### Quick Reference Patterns

-   **Pagination**: `limit=20&cursor=xyz` (Better than Offset).
-   **Filters**: `status=active&created_gt=2023-01-01`.
-   **Fields**: `fields=id,name,email` (Partial response).

// end-parallel

---

## Quality Assurance

### Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| Verbs in URL | `/createUser` -> `POST /users`. |
| 200 OK for Error | Use 400/500 codes. Don't hide errors in JSON body. |
| Breaking Change | Ensure backward compatibility. Add `@deprecated`. |
| N+1 (GraphQL) | Use DataLoader pattern. |

### API Checklist

- [ ] Plural nouns for resources
- [ ] Proper HTTP Status Codes
- [ ] Versioning strategy defined
- [ ] Pagination (Cursor preference)
- [ ] Error objects standardized
- [ ] Documentation (OpenAPI/Swagger)
