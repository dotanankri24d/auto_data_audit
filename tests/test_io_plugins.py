import io
import json
import zipfile
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from data_audit import (load_csv,load_excel,list_excel_sheets,map_columns,audit,NeighborAudit,load_run,
                        bind_policy,ReviewPolicy,ValidationError)
from data_audit.plugins import discover_plugins,parameter_schema


def test_csv_id_bom_headers_structure_and_mapping():
    raw='\ufeffid,x,label,meta\n001,1,A,hi\n002,2,B,bye\n'.encode()
    f=load_csv(io.BytesIO(raw),id_column='id')
    assert list(f.index)==['001','002']
    data=map_columns(f,label='label',features=['x'],metadata=['meta'])
    r=audit(data.X,data.y,metadata=data.metadata,methods=[NeighborAudit(n_neighbors=1,radius_mode='none')])
    assert b'001' in r.export_csv()
    for content,match in [('a,a\n1,2\n','duplicate'),('a,b\n1\n','source row 2'),('a,id\n1,\n','missing ID')]:
        with pytest.raises(ValidationError,match=match):
            load_csv(io.BytesIO(content.encode()),id_column='id' if 'id' in content else None)
    with pytest.raises(ValidationError,match='cannot be selected'):
        map_columns(f,label='label',features=['label'])
    bad=load_csv(io.BytesIO(b'id,x,label\n001,oops,A\n'),id_column='id')
    with pytest.raises(ValidationError,match="source row 2, column 'x'.*oops"):
        map_columns(bad,label='label',features=['x'])
    with pytest.raises(ValidationError,match='source row 2'):
        load_csv(io.BytesIO(b'id,x\n001,oops\n'),id_column='id',dtype={'x':float})
    with pytest.raises(ValidationError,match='Local file'):
        load_csv('https://example.com/file.csv')


@pytest.mark.parametrize('extension',['xlsx','xls'])
def test_excel_readers(extension,tmp_path):
    f=pd.DataFrame({'id':['001','002'],'x':[1,2],'label':['A','B']})
    path=tmp_path/('data.'+extension)
    if extension=='xlsx':
        f.to_excel(path,index=False,sheet_name='נתונים')
    else:
        import xlwt
        w=xlwt.Workbook()
        s=w.add_sheet('נתונים')
        for i,row in enumerate([list(f.columns),*f.values.tolist()]):
            for j,value in enumerate(row):
                s.write(i,j,value)
        w.save(str(path))
    assert list_excel_sheets(path)==['נתונים']
    loaded=load_excel(path,id_column='id')
    csv=load_csv(io.BytesIO(f.to_csv(index=False).encode()),id_column='id')
    assert_frame_equal(loaded,csv)
    assert loaded.index.tolist()==['001','002']


def test_excel_multisheet_duplicate_and_conversion(tmp_path):
    path=tmp_path/'multi.xlsx'
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({'id':['01'],'x':['oops']}).to_excel(writer,sheet_name='one',index=False)
        pd.DataFrame({'id':['02'],'x':[2]}).to_excel(writer,sheet_name='two',index=False)
    with pytest.raises(ValidationError,match="one.*two"):
        load_excel(path)
    with pytest.raises(ValidationError,match='not found'):
        load_excel(path,sheet_name='missing')
    with pytest.raises(ValidationError,match="one: source row 2, column 'x'.*oops"):
        load_excel(path,sheet_name='one',id_column='id',dtype={'x':float})
    import openpyxl
    w=openpyxl.Workbook()
    w.active.append(['x','x'])
    w.active.append([1,2])
    w.save(path)
    with pytest.raises(ValidationError,match='duplicate'):
        load_excel(path)


def test_external_extensions_schema_refresh_and_output(tmp_path):
    for kind in ['embeddings','methods','policies']:
        (tmp_path/kind).mkdir()
    embedding=tmp_path/'embeddings'/'normalize.py'
    embedding.write_text('import numpy as np\ndef fit(X, epsilon=1e-8):\n'
                         '    if epsilon <= 0: raise ValueError("epsilon must be positive")\n'
                         '    mean=X.mean(0); scale=np.maximum(X.std(0),epsilon)\n'
                         '    return lambda x:(x-mean)/scale\n')
    (tmp_path/'methods'/'center.py').write_text('import pandas as pd\nimport numpy as np\n'
        'def analyze(data, power=2.0):\n'
        '    z=data.fit_embedding(np.arange(len(data.X)))(data.X)\n'
        '    return {"samples":pd.DataFrame({"sample_id":data.sample_ids,"distance":np.linalg.norm(z,axis=1)**power})}\n')
    (tmp_path/'policies'/'threshold.py').write_text('import pandas as pd\n'
        'def decide(samples, threshold=0.7):\n'
        '    return pd.DataFrame({"review":samples["center.distance"].ge(threshold).astype("boolean")},index=samples.index)\n')
    plugins,errors=discover_plugins(tmp_path)
    assert not errors
    by={p.kind:p for p in plugins}
    assert by['embedding'].fields[0].name=='epsilon'
    r=audit([[0,1],[1,0],[2,1]],['A','B','A'],embedding=by['embedding'].bind(),methods=[by['method'].bind()])
    assert len(r.apply_policy('center',by['policy'].bind()).flags)==3
    old=by['embedding'].source_hash
    embedding.write_text(embedding.read_text().replace('epsilon=1e-8','epsilon=1e-8, enabled=True'))
    refreshed,errors=discover_plugins(tmp_path)
    p=next(p for p in refreshed if p.kind=='embedding')
    assert p.source_hash!=old and len(p.fields)==2
    (tmp_path/'methods'/'broken.py').write_text('def analyze(data, *args): return {}')
    _,errors=discover_plugins(tmp_path)
    assert 'open' in str(errors)
    # Saved results can be viewed after extension code has been removed.
    raw=r.save()
    embedding.unlink()
    q=load_run(io.BytesIO(raw))
    assert_frame_equal(q.samples,r.samples)


def test_signature_diagnostics():
    from typing import Literal
    def fields(data, required:int, enabled=True, mode:Literal['a','b']='a', option:float|None=None):
        pass
    schema=parameter_schema(fields,1)
    assert schema[0].required and schema[-1].nullable
    assert schema[2].choices==('a','b')
    def ambiguous(data,thing=None):
        pass
    def unsupported(data,thing:list=[]):
        pass
    for fn in [ambiguous,unsupported]:
        with pytest.raises(ValidationError):
            parameter_schema(fn,1)
    def policy(samples,threshold=0.7):
        if not 0<=threshold<=1:
            raise ValueError('threshold must be between 0 and 1')
        return pd.DataFrame({'flag':pd.Series(pd.NA,index=samples.index,dtype='boolean')})
    r=audit([[0],[1]],['A','B'],methods=[NeighborAudit(n_neighbors=1,radius_mode='none')])
    with pytest.raises(ValueError,match='between'):
        r.apply_policy('neighbors',bind_policy(policy,threshold=2))
    assert r.apply_policy('neighbors',ReviewPolicy()).flags.isna().all().all()


def test_bundle_rejects_unknown_schema_and_invalid_data():
    with pytest.raises(ValidationError,match='Invalid audit bundle'):
        load_run(io.BytesIO(b'not zip'))
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w') as z:
        z.writestr('manifest.json',json.dumps({'schema_version':'999'}))
    with pytest.raises(ValidationError,match='Unsupported bundle schema'):
        load_run(io.BytesIO(out.getvalue()))
