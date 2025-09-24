# AI Code Assistant Task Execution Prompt

## COMMAND: ANALYZE -> PLAN -> EXECUTE -> TRACK

### Your Mission
Review the entire codebase systematically, create a detailed action plan, execute it step by step, and maintain real-time progress tracking.

### IMMEDIATE ACTION: Process User Request
If the user provides specific tasks after this command:
1. Add them to projectToDos.md under "User Requested - HIGH PRIORITY"
2. These tasks take precedence over ALL existing tasks
3. Start working on them immediately after analysis
4. Mark clearly as "USER REQUEST - [date]"

### Required Actions

#### 1. ANALYZE (First 5 Minutes)
- Read CODEBASE_DOCUMENTATION.md completely
- Read projectToDos.md completely
- Scan key source files to understand architecture
- Identify dependencies and critical paths

#### 2. PLAN (Next 5 Minutes)
- Pick the highest priority incomplete task from projectToDos.md
- Break it into 5-10 concrete steps
- Identify files that need modification
- List tests that need to be written
- Update projectToDos.md with your plan

#### 3. EXECUTE (Core Work)
For each step in your plan:
- Mark task as "in_progress" in projectToDos.md
- Write the code/fix/feature
- Test it immediately
- Mark as "completed" in projectToDos.md
- Move to next step

#### 4. TRACK (Continuous)
- Update projectToDos.md after EVERY file change
- Add new discovered issues immediately
- Move completed items to "Completed" section with date
- Keep task list current and accurate

### Progress Update Format

After each significant change, update projectToDos.md like this:

```markdown
## USER REQUESTED - HIGH PRIORITY
### [Task from user]
Status: IN PROGRESS
Requested: [date/time]
Priority: IMMEDIATE

## Active Development

### Current Task: [Task Name]
Status: IN PROGRESS
Started: [timestamp]

Steps Completed:
- [x] Step 1 description
- [x] Step 2 description
- [ ] Step 3 description (current)
- [ ] Step 4 description

Files Modified:
- path/to/file1.py (lines 10-50)
- path/to/file2.py (lines 100-150)

Next Action: [specific next step]
```

### Rules

1. NO ASSUMPTIONS - Read the actual code first
2. NO PLACEHOLDERS - Write real, working code
3. TEST EVERYTHING - Run your code before marking complete
4. DOCUMENT CHANGES - Update docs as you go
5. SMALL COMMITS - One feature per commit

### Priority Order

1. USER REQUESTED TASKS (from this session)
2. Critical bugs (crashes, data loss)
3. Performance issues (>1 second delays)
4. User-facing features
5. Backend improvements
6. Documentation
7. Nice-to-haves

### Success Metrics

Your work is complete when:
- [ ] Task from projectToDos.md is fully implemented
- [ ] All tests pass
- [ ] Documentation updated
- [ ] projectToDos.md shows accurate status
- [ ] Code committed with clear message

### Emergency Protocol

If you encounter blockers:
1. Document the exact error in projectToDos.md
2. Try alternative approach
3. If still blocked, mark task as "BLOCKED" with reason
4. Move to next priority task
5. Return to blocked task later

### Final Checklist

Before marking any task complete:
- [ ] Code works end-to-end
- [ ] No commented-out code remains
- [ ] No debug prints remain
- [ ] projectToDos.md is current
- [ ] Tests cover the changes

---

## START NOW

1. Check for user-specified tasks in this request
2. Add any user tasks to projectToDos.md as HIGH PRIORITY
3. Open CODEBASE_DOCUMENTATION.md
4. Open projectToDos.md
5. Start with user tasks (if any), otherwise pick from existing
6. Begin execution
7. Update progress every 10 minutes

Remember: USER TASKS FIRST. SHIP WORKING CODE. TRACK EVERYTHING. NO EXCUSES.