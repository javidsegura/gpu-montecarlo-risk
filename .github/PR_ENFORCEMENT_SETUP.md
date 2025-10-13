# PR Template Enforcement Setup Guide

This guide explains how to enforce the PR templates and requirements for the GPU-accelerated Monte Carlo project.

## 🚀 Quick Setup (GitHub Repository Settings)

### 1. Enable Branch Protection Rules

Go to **Settings > Branches** and create protection rules:

#### For `main` branch:
- ✅ **Require a pull request before merging**
- ✅ **Require approvals**: 2 reviewers
- ✅ **Dismiss stale PR approvals when new commits are pushed**
- ✅ **Require review from code owners**
- ✅ **Restrict pushes that create files larger than 100MB**
- ✅ **Require status checks to pass before merging**:
  - `PR Validation`
  - `Code Quality Checks`
  - `Build and Test`
  - `Performance Impact Check` (for performance PRs)
- ✅ **Require branches to be up to date before merging**
- ❌ **Allow force pushes**
- ❌ **Allow deletions**

#### For `develop` branch:
- ✅ **Require a pull request before merging**
- ✅ **Require approvals**: 1 reviewer
- ✅ **Dismiss stale PR approvals when new commits are pushed**
- ✅ **Require status checks to pass before merging**:
  - `PR Validation`
  - `Code Quality Checks`
  - `Build and Test`

### 2. Configure Status Checks

The following status checks will be automatically enforced:

#### `PR Validation`
- Validates PR template completion
- Checks for required sections
- Verifies performance impact documentation
- Ensures test results are provided

#### `Code Quality Checks`
- Static analysis with cppcheck
- TODO/FIXME detection in code
- Code style validation

#### `Build and Test`
- Multi-compiler builds (GCC, Clang)
- Multi-MPI builds (OpenMPI, MPICH)
- Unit and integration tests
- CUDA compilation verification

#### `Performance Impact Check` (conditional)
- Only runs for performance-related PRs
- Validates performance metrics documentation
- Ensures benchmark results are provided

### 3. Set Up Code Owners

The `CODEOWNERS` file automatically assigns reviewers based on file paths:
- Core algorithms: Requires lead developer review
- Performance code (CUDA/MPI/OpenMP): Requires performance team review
- Documentation: Requires documentation team review

## 🔧 Manual Configuration Steps

### Step 1: Repository Settings
1. Go to your GitHub repository
2. Click **Settings** tab
3. Navigate to **Branches**
4. Click **Add rule** or **Add branch protection rule**

### Step 2: Configure Branch Protection
1. **Branch name pattern**: `main`
2. **Protect matching branches**: ✅
3. **Require a pull request before merging**: ✅
4. **Require approvals**: Set to 2
5. **Dismiss stale PR approvals when new commits are pushed**: ✅
6. **Require review from code owners**: ✅

### Step 3: Configure Status Checks
1. **Require status checks to pass before merging**: ✅
2. **Require branches to be up to date before merging**: ✅
3. **Status checks**: Select the following:
   - `PR Validation`
   - `Code Quality Checks`
   - `Build and Test`

### Step 4: Configure Restrictions
1. **Restrict pushes that create files larger than 100MB**: ✅
2. **Allow force pushes**: ❌
3. **Allow deletions**: ❌

### Step 5: Save Configuration
Click **Create** to save the branch protection rule.

## 📋 PR Template Requirements

### Mandatory Sections
Every PR must include:
- **Description**: Clear explanation of changes
- **Type of Change**: Bug fix, feature, performance improvement, etc.
- **Technical Details**: Implementation approach (MPI/OpenMP/CUDA)
- **Testing**: Test coverage and results
- **Performance Impact**: Speedup, memory usage, scalability

### Required Information
- **Performance Impact**: Must specify expected speedup and memory usage
- **Test Results**: Must include test execution results
- **Technical Approach**: Must specify which parallelization techniques are used
- **Checklist**: All checkboxes must be completed

### Automated Validations
The system automatically checks for:
- Presence of required sections
- Completion of mandatory fields
- Test result documentation
- Performance impact specification

## 🚨 Enforcement Rules

### PR Rejection Criteria
PRs will be automatically blocked if:
- Missing required template sections
- No performance impact documented
- No test results provided
- Unchecked required checkboxes
- Failing status checks

### Review Requirements
- **Core algorithms**: 2 approvals + code owner review
- **Performance changes**: 2 approvals + performance team review
- **Documentation**: 1 approval
- **Bug fixes**: 1 approval + code owner review

### Status Check Failures
- **Build failures**: PR cannot be merged
- **Test failures**: PR cannot be merged
- **Code quality issues**: PR cannot be merged
- **Missing performance data**: PR cannot be merged (for performance PRs)

## 🔍 Monitoring and Compliance

### PR Dashboard
Monitor PR compliance through:
- GitHub PR list with status check indicators
- Required checks section in each PR
- Review status and approval requirements

### Compliance Reports
- Weekly PR compliance reports
- Performance impact tracking
- Code quality metrics
- Review cycle time analysis

## 🛠️ Troubleshooting

### Common Issues

#### "PR Validation Failed"
- Check that all required sections are present
- Verify performance impact is documented
- Ensure test results are provided

#### "Code Quality Checks Failed"
- Run `cppcheck` locally to identify issues
- Remove or fix TODO/FIXME comments
- Address static analysis warnings

#### "Build and Test Failed"
- Verify code compiles with required compilers
- Ensure all tests pass locally
- Check CUDA compilation if applicable

#### "Performance Impact Check Failed"
- Document expected performance changes
- Include benchmark results
- Specify memory usage impact

### Getting Help
- Check GitHub Actions logs for detailed error messages
- Review the PR template requirements
- Contact team leads for clarification on requirements

## 📈 Benefits

### For Contributors
- Clear expectations and requirements
- Automated feedback on PR quality
- Reduced review cycles through better preparation
- Consistent documentation standards

### For Maintainers
- Higher quality contributions
- Reduced review overhead
- Better performance tracking
- Consistent code standards

### For the Project
- Improved code quality
- Better performance documentation
- Faster development cycles
- Enhanced maintainability

---

**Note**: These enforcement rules ensure high-quality contributions while maintaining development velocity. All rules are designed to be practical and achievable for HPC development workflows.
