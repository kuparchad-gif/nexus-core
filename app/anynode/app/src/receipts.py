import json, time, uuid, os
LOG=os.getenv('RECEIPTS_LOG','receipts.jsonl')
def write(caller,tool,args,res,lineage_id=None):
  rec={'id':f'rcpt:{uuid.uuid4()}','ts':time.time(),'caller':caller,'tool':tool,'args':args,'result':res,'lineage_id':lineage_id}
  with open(LOG,'a',encoding='utf-8') as f: f.write(json.dumps(rec)+'\n')
  return rec
