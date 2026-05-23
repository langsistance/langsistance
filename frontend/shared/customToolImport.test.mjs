import test from 'node:test';
import assert from 'node:assert/strict';

import { buildCustomToolImportPayload } from './customToolImport.js';

test('wraps custom tool JSON for create_tool_from_custom', () => {
  const spec = {
    url: 'https://api.uspto.gov/api/v1/patent/applications/search',
    method: 'POST',
    query: {},
    body: {
      q: 'applicationNumberText:"18244278"',
    },
  };

  const payload = buildCustomToolImportPayload(spec);

  assert.deepEqual(payload, {
    spec_format: 'json',
    spec_content: JSON.stringify(spec),
  });
});
