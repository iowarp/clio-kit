import {execFileSync} from 'node:child_process';

// CLIO_PLUGIN_ROOT is supplied by Kit's project installer.
export default async ({directory}) => ({
  'tool.execute.after': async (input, output) => {
    if (!['write', 'edit'].includes(input.tool)) return;
    const path = input.args?.filePath;
    if (!path) return;
    const response = execFileSync('python3', [
      `${CLIO_PLUGIN_ROOT}/skills/dataset-report/scripts/verify_report.py`, 'hook',
    ], {
      input: JSON.stringify({cwd: directory, tool_input: {file_path: path}}),
      encoding: 'utf8', timeout: 25000,
    }).trim();
    if (response) {
      output.output += `\n${JSON.parse(response).hookSpecificOutput.additionalContext}`;
    }
  },
});
