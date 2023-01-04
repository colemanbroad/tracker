.separator ";"
create Table tab(Info,Skip,Time,OnOff,Name);
.import logfile.csv tab
.mode box
SELECT * FROM tab ; 
-- SELECT * FROM tab WHERE Name LIKE "%init%" ;
SELECT Name , (Time - LAG(Time) OVER ()) / 1000.0 Delta FROM tab ;
