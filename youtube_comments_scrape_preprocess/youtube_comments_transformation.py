import dateparser as dateparser
import pandas as pd
import timedelta
import datetime as datetime, timedelta
import dateparser

df = pd.read_csv('/Users/lidiiamelnyk/Documents/tweets_tsg/Youtube .csv', sep = ';')

df['date'] = df['comment_time'].apply(lambda x: dateparser.parse(x.replace('(edited)','').strip(''), settings={'RELATIVE_BASE': datetime.datetime(datetime.now())}).strftime('%m/%d/%Y'))
df = df.drop_duplicates(subset ="content")
df = df[len(df['content'])>3]
new_columns = 'video_url', 'comment_user', 'content', 'date'
df.reindex(columns = new_columns)

with open('/Users/lidiiamelnyk/Documents/tweets_tsg/youtube_corrected.csv', 'w+', newline = '', encoding='utf-8-sig') as file:
    df.to_csv(file, sep=';', na_rep='', float_format=None,
               columns= new_columns,
               header=True, index=False, index_label=None,
               mode='a', compression='infer',
               quoting=None, quotechar='"', line_terminator=None, chunksize=None,
               date_format=str, doublequote=True, escapechar=None, decimal='.', errors='strict')
    file.close()
