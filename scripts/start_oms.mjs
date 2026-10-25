// Установленный upstream API без изменения исходников; bind только loopback.
import {createServer} from 'node:http';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
const root = resolve('.runtime/analogs/oms');
process.chdir(root);
const load = name => import(pathToFileURL(resolve(root, 'src', name)).href);
const {loadConfig} = await load('config.js');
const {createApp} = await load('server/createApp.js');
const config = loadConfig({...process.env, PORT:'8787', TRANSCRIPTION_PROVIDER:'mock',
  NOTE_PROVIDER:'openai', OPENAI_API_KEY:'local-only',
  OPENAI_BASE_URL:process.env.LOCAL_MODEL_URL || 'http://127.0.0.1:8082',
  OPENAI_NOTE_MODEL:'medical-local', REDACT_BEFORE_API_CALLS:'false'});
const app = createApp({config});
createServer(app.handler).listen(8787,'127.0.0.1',()=>console.log('OMS API 127.0.0.1:8787 → '+config.openai.baseUrl));
