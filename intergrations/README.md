# TQB++ Integrations

This directory hosts various integrations for the Tiny QA Benchmark++ (TQB++) project.

## Available Integrations

### [OpenAI Evals](./openai-evals/README.md)
Configuration files and instructions for running evaluations using the OpenAI Evals framework. This integration allows you to:
- Run TQB++ datasets through OpenAI's evaluation framework
- Compare model performance using standardized metrics
- Generate detailed evaluation reports

### [Opik](./opik/README.md)
Integration with [Opik](https://github.com/comet-ml/opik) for dataset management and evaluation. This integration enables you to:
- Generate synthetic QA pairs using TQB++
- Create and manage datasets through Opik's interface
- Track and version your evaluation datasets
- Access datasets through Opik's web interface

## Adding New Integrations

To add a new integration:
1. Create a new directory under `integrations/`
2. Include a `README.md` with:
   - Overview of the integration
   - Setup instructions
   - Usage examples
   - Configuration options
3. Add any necessary code files
4. Update this README with the new integration details

## Integration Guidelines

When creating new integrations:
- Follow the existing directory structure
- Document all configuration options
- Include example usage
- Provide clear setup instructions
- Add appropriate error handling
- Include any necessary dependencies in requirements files
