Postup
------

1. www.openstreetmap.org
   prave tlacitko mysi - Query Features dava gps polohu a dalsie info o bode
   prave tlacitko mysi - Show address dava link z ktoreho sa da prevziat z mapy vypocitana poloha 

2. zvolime tab Export, zadame manually borders, exportujeme do map.osm

3. manualne vytvorime range.xml podla udajov z map.osm a pripadne doladime cez calib.py (TBD)

4. pomocou borders.py vytvorime area.xml vybratim polygomu bodov ktore uz do 
   sutaznej oblasti nepatria
   
5. pomocou mapper.py zobrazime cesty

6. pomocou tracer.py na mape vieme zobrazit zaznam z gps.txt (TBD)
