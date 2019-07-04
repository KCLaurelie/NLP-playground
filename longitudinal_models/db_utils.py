# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 09:00:22 2019

@author: AMascio
"""

import pandas as pd
import pandas.io.sql as psql
import pyodbc
import sys
import sqlalchemy as sa
import os

_server_name = 'BRHNSQL094'
_db_name = 'SQLCRIS'
root_path = r'T:\aurelie_mascio'
CRIS_data_path = root_path + '\\CRIS data'
SQL_path = root_path + '\\SQL queries'


def connect(server_name=_server_name, db_name=_db_name):
    connection_string = 'Driver={SQL Server Native Client 10.0};Server=' + server_name + ';Database=' + db_name + ';Trusted_Connection=yes;'
    connection = pyodbc.connect(connection_string)
    return connection


def query_sql_alchemy(query, server_name=_server_name, db_name=_db_name):
    connection_string = 'Driver={SQL Server Native Client 10.0};Server=' + server_name + ';Database=' + db_name + ';Trusted_Connection=yes;'
    engine = sa.create_engine("mssql+pyodbc:///?odbc_connect={}".format(connection_string))

    if '.sql' in query:
        fd = open(query, 'r')
        sqlFile = fd.read()
        fd.close()
        sqlCommand = sqlFile.split(';')[0]  # only take 1st query
    else:
        sqlCommand = query
    # query="select top 10 * from [SQLCRIS].[dbo].Attachment"
    sql_reader = pd.read_sql(sqlCommand, con=engine, chunksize=10000)
    res = pd.Dataframe()
    for sql_chunk in sql_reader:
        res = pd.concat([res, sql_chunk])

    return res


def list_table_names(server_name=_server_name, db_name=_db_name):
    query = """SELECT TABLE_NAME
  FROM INFORMATION_SCHEMA.TABLES
  WHERE TABLE_TYPE = 'BASE TABLE' AND TABLE_CATALOG='{}'""".format(db_name)

    connection = connect(server_name, db_name)
    df = psql.read_sql(query, connection)
    connection.close()

    return df['TABLE_NAME'].tolist()


def list_column_names(table_name, server_name=_server_name, db_name=_db_name, verbose=False):
    query = """SELECT *
  FROM INFORMATION_SCHEMA.COLUMNS
  WHERE TABLE_NAME = '{}'
  AND TABLE_SCHEMA='dbo'""".format(table_name)

    query = """SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, ORDINAL_POSITION
  FROM INFORMATION_SCHEMA.COLUMNS
  WHERE TABLE_NAME = '{}'
  ORDER BY TABLE_NAME, ORDINAL_POSITION""".format(table_name)

    if verbose:
        print('Query:', query)

    connection = connect(server_name=_server_name, db_name=_db_name)
    df = psql.read_sql(query, connection)
    connection.close()

    return df['COLUMN_NAME'].tolist()


def fetch_dataframe(query, server_name=_server_name, db_name=_db_name):
    print('-- Fetching DataFrame. Server:', server_name, 'Database:', db_name, file=sys.stderr)
    df = pd.DataFrame()

    try:
        connection = connect(server_name=_server_name, db_name=_db_name)
        df = psql.read_sql(query, connection)
        connection.close()
    except Exception as e:
        print("-- Error accessing database", file=sys.stderr)
        print(str(e), file=sys.stderr)

    return df


# extract_corpus_for_brcid_list(sql_query=all_attachments_sql,output_path=root_path+r'\multimorbidity\corpus', brcid_source=root_path+r'\multimorbidity\brcid_list.csv')
def extract_corpus_for_brcid_list(sql_query,
                                  server_name=_server_name, db_name=_db_name,
                                  output_path=root_path + r'\F20 corpus\training',
                                  brcid_source=CRIS_data_path + r'\F20_brcid_training.csv',
                                  start_date='2007-01-01'):
    all_brcids = pd.read_csv(brcid_source, header=0, chunksize=2000, usecols=['brcid'])
    connection = connect(server_name=_server_name, db_name=_db_name)
    for chunk in all_brcids:
        tmp_brc_ids = ",".join(map(str, list(chunk.brcid.unique())))
        #        formatted_sql_query = """ select distinct convert(varchar,att.Date,23) as 'Date',CN_Doc_ID, BrcId,Attachment_Text
        #        from [SQLCRIS].[dbo].Attachment att
        #        where BrcId in ({})
        #        and Attachment_Text is not null and Attachment_Text!='' and DATALENGTH( Attachment_Text)>1
        #        """.format(tmp_brc_ids)
        formatted_sql_query = sql_query.format(start_date, tmp_brc_ids, tmp_brc_ids, tmp_brc_ids, tmp_brc_ids)
        query_res = psql.read_sql(formatted_sql_query, connection)
        for record in query_res.itertuples():
            if len(record.Attachment_Text) > 1:  # sometimes sql query doesn't exclude all empty texts...
                print('generating corpus for patient', record.BrcId)
                title = str(record.BrcId) + '_' + str(record.Date) + '_' + str(record.CN_Doc_ID)
                tmp_path = output_path + '\\' + str(record.BrcId)
                if not os.path.exists(tmp_path): os.makedirs(tmp_path)
                with open(os.path.join(tmp_path, title + '.txt'), "w") as text_file:
                    try:
                        text_file.write(record.Attachment_Text)  # sometimes it breaks because of weird characters
                    except:
                        pass  # in which case we skip for now (TODO: clean weird stuff in SQL query)
    connection.close()
    return 0


def execute_sql(SQL_file,
                server_name=_server_name,
                db_name=_db_name,
                save_to_csv=False):
    res = pd.DataFrame()
    fd = open(SQL_file, 'r')
    sqlFile = fd.read()
    fd.close()

    # all SQL commands (split on ';')
    sqlCommands = sqlFile.split(';')

    # Execute every command from the input file
    for command in sqlCommands:
        res = pd.concat([res, fetch_dataframe(server_name, db_name, str(command))])
    if save_to_csv:
        res.to_csv(SQL_file.replace('.sql', '.csv'))
        return 'file saved under: ' + SQL_file.replace('.sql', '.csv')
    return res


# test=execute_sql(SQL_file=SQL_path+'\\F20_all_texts.sql')
# test=test.sort_values(by=['BrcId', 'date'])
# test.to_csv(SQL_path+'\\F20_all_texts.sql')

all_attachments_sql = """
DECLARE @min_diagnosis_date DATETIME='{}';
WITH attachments as 
(
select distinct
  att.BrcId, cast(att.CN_Doc_ID as bigint) as CN_Doc_ID, convert(varchar,att.Date,23) as doc_date, att.Attachment_Text, 'attachment' as doc_type
  FROM SQLCRIS.dbo.Attachment att 
  where DATALENGTH(att.Attachment_Text)>0 and att.Date > DATEADD(month,-6,@min_diagnosis_date)
  and BrcId in ({})
 )
,discharge_docs as 
(
select distinct dc.BrcId, cast(dc.CN_Doc_ID as bigint) as CN_Doc_ID, convert(varchar,dc.Completion_Date,23) as doc_date, dc.Brief_Summary as Attachment_Text, 'discharge_summary' as doc_type
    from SQLCRIS.dbo.Discharge_Notification_Summary dc 
    where DATALENGTH(dc.Brief_Summary)>0 and dc.Completion_Date > DATEADD(month,-6,@min_diagnosis_date)
    and BrcId in ({})
) 
,discharge_plans as 
(
select distinct dc.BrcId, cast(dc.CN_Doc_ID as bigint) as CN_Doc_ID, convert(varchar,dc.Completion_Date,23) as doc_date, dc.Discharge_Plan as Attachment_Text, 'discharge_plan' as doc_type
    from SQLCRIS.dbo.Discharge_Notification_Summary dc 
    where DATALENGTH(dc.Brief_Summary)>0 and dc.Completion_Date > DATEADD(month,-6,@min_diagnosis_date)
    and BrcId in ({})
)
,correspondences as 
(
select distinct cc.BrcId, cast(cc.CN_Doc_ID as bigint) as CN_Doc_ID, convert(varchar,cc.Date,23) as doc_date,cc.Attachment_Text, 'correspondence' as doc_type
    from SQLCRIS.dbo.CCS_Correspondence cc 
    where DATALENGTH(cc.Attachment_Text)>0 and cc.Date > DATEADD(month,-6,@min_diagnosis_date)
    and BrcId in ({})
) 

select * from  attachments UNION select * from discharge_docs union select * from discharge_plans union select * from correspondences

"""
