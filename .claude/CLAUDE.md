# CLAUDE.md - Development Standards

## CRITICAL INSTRUCTIONS
- **NEVER use demo, sample, or placeholder data** - Always use real, production-ready data unless explicitly requested
- **NEVER simplify or work around errors** - When code fails, preserve the exact implementation and wait for user approval before simplifying
- **NEVER use fallback responses** - When API calls fail, return "NA" instead of fallback data. NO EXCEPTIONS!
- **Follow these instructions exactly** - Do not deviate from these standards
- **NEVER Claim completion of job or stop working until key outcome as defined by user at the start is achieved.  Keep working.  Keep debugging. Keep testing until outcome is achieved with FULL FUNCTIONALITY AS REQUESTED and no simplificaitons**
- **If you think you have completed and achieved the outcome, you need to provide data and proof eg date modified for file created, result of a database query showing populated records etc etc.  Do NOT CLAIM COMPLETION WITHOUT PROOF!**
**Never use unicode characters.  Not in omments, debug or conversation.  EVER!**


## Development Process

### Before Writing Code
- Fully understand requirements before implementation
- Ask clarifying questions if specifications are unclear
- Identify existing patterns in codebase to maintain consistency

### Implementation Standards
- Write production-ready code on first attempt
- Include comprehensive error handling
- Follow existing project patterns and conventions
- Maintain consistent naming across the codebase
- Keep functions small and focused (single responsibility)

### Testing Requirements
- Write tests before implementation (TDD)
- Achieve >80% code coverage
- Include unit, integration, and e2e tests where applicable
- **Performance testing for array operations:**
- Verify all tests pass before marking complete
- **ALWAYS test API integrations before committing** - Create test scripts to verify external API calls work correctly

### Code Quality Checklist
Before completing any task, verify:
- [ ] All tests passing
- [ ] Code follows project conventions
- [ ] Error handling implemented
- [ ] Performance optimized
- [ ] Documentation complete
- [ ] No console.logs or debug code remains

## Technical Standards

### Array Processing & Performance
- Use vectorized operations (NumPy, Pandas, native array methods)
- Avoid nested loops for data processing
- Implement sliding window operations efficiently
- Pre-allocate arrays when size is known
- Use appropriate data structures (Set/Map for lookups vs arrays)
- Profile before optimizing unclear bottlenecks

### Code Organization
- Follow DRY (Don't Repeat Yourself) principle
- Apply SOLID principles where appropriate
- Use clear, self-documenting variable and function names
- Organize imports logically (external deps, internal modules, types)

### Documentation
 - Create/ update/ append a CODE DOCUMENTATION.md in this directory with a complete code overview including diagrams, directory structure, functional summary per module at EVERY COMMIT.  This file should be detailed enough for an experienced coder to get a good idea of where all the code fits into the architecture and functionality of the codebase
- Document all public APIs
- Include JSDoc comments for complex functions
- Update README when adding new features
- Keep inline comments minimal and meaningful

### Version Control
- Write clear, descriptive commit messages
- One logical change per commit
- Reference issue numbers where applicable

## Project-Specific Configuration

### Commands
```bash
# Development
npm run dev          # Start development server
npm run build        # Build for production
npm run test         # Run all tests
npm run test:watch   # Run tests in watch mode
npm run lint         # Check code style
npm run format       # Auto-format code
```

## Working Instructions

### When Encountering Errors
1. Report the exact error message
2. Maintain current approach unless directed otherwise
3. Wait for user guidance before simplifying
4. Never automatically fall back to simpler solutions


### Code Review Protocol
1. Implement functionality completely
2. Run all tests and linting
3. Review against quality checklist
4. Report completion status with any issues found

### Continuous Improvement
- Update this file when new patterns are established
- Remove outdated instructions
- Keep instructions specific and actionable

## Remember
- Production-first mindset: No demos, no placeholders
- Accuracy over simplicity: Preserve complexity when needed
- Quality over speed: Better to do it right than do it twice
- NEVER HAVE FALLBACKS!!!!  Have NA if there is no data.  Please add to CLAUDE.md