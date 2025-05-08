# API Reference

This section provides detailed documentation for all AImPact platform APIs.

## Authentication

All API endpoints (except auth endpoints) require a valid JWT token in the Authorization header:

```
Authorization: Bearer {token}
```

### Response Format

All API responses follow a standard format:

```json
{
  "success": true,
  "data": {
    // Response data here
  },
  "error": null,
  "meta": {
    "version": "1.0",
    "timestamp": "2025-05-09T00:14:30.116670+03:00"
  }
}
```

Error responses:

```json
{
  "success": false,
  "data": null,
  "error": {
    "code": "ERROR_CODE",
    "message": "Error description",
    "details": {}
  },
  "meta": {
    "version": "1.0",
    "timestamp": "2025-05-09T00:14:30.116670+03:00"
  }
}
```

### Rate Limiting

APIs implement rate limiting based on the user's subscription plan:

| Plan | Rate Limit |
|------|------------|
| Free | 100 req/hour |
| Basic | 1,000 req/hour |
| Pro | 10,000 req/hour |
| Enterprise | Custom |

Rate limit headers are included in all responses:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1620000000
```

## API Service Documentation

Each service has its own dedicated API documentation:

- [Auth API](./auth.md)
- [Voice API](./voice.md)
- [Agent API](./agents.md)
- [Workflow API](./workflows.md)
- [Memory API](./memory.md)
- [Optimizer API](./optimizer.md)
- [Recommendation API](./recommendations.md)

## Versioning

The API uses URI versioning:

```
/api/v1/resource
```

## OpenAPI Specification

The full OpenAPI specification is available at `/api/openapi.json` or through the Swagger UI at `/api/docs`.

