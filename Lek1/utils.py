def get_minutes(hour: str) -> int:
    '''
    Ta in tidsdata från flights_data, ge tillbaks antal minuter från 00:00.
    get_minutes('6:13') -> 373
    get_minutes('23:59') -> 1439
    get_minutes('00:00') -> 0
    '''
    pass



#Ladda in flight_data.json, omvandla från {"BRU-FCO": [["6:12", "10:22", 230],...} till 
#{("BRU", "FCO"): [["6:12", "10:22", 230],..}
#Varje sublista för (Startplats, Mål) innehåller [avgångstid, landningstid, biljettpris]


#Flight_data innehåller även tuples av (Namn, Hemstad) - t.ex. (London , LHR) där London är Londonpersonens namn och LHR markerar 
#origin. 
#Sist i filen är destination.





def print_schedule(schedule):
    pass
#Skriv en funktion som tar in listan schedule med 2*n_people integers x_n, där 0<= x_n <= 9.
#Fyll en lista med slumpade integers i denna range och slussa in!

# För listan [1,3,6,2...] med 2*n_people element ska funktionen printa för person 0 personens namn, 
# startplats, avgångstid, ankomsttid och biljettpris
# från index 1 i avgångar för (Person0_Origin, Destination), följt av startplats, avgångstid, ankomsttid från index 3 
# i avgångar för (Destination, Person0_Origin).

#Funktionen ska sedan printa totala priset för alla avgångar


def fitness_function(solution):
    pass
#Vi vill minimera tid och pengar (totalt biljettpris och kumulativ väntetid för våra resenärer i filen)
#Alla personer ska alltså anlända i FCO och åka därifrån samtidigt. När den sista personen har anlänt slutar klockan ticka.
#

#Definiera fitnessfunktionen som total väntetid + totalt biljettpris och returnera detta värde för jämförelse för optimeringsfunktioner.



def random_search(domain, fitness_function):
    pass
#Som sökdomän / utfallsrum / lösningsrum ska vi alltså kunna stoppa in vilka index för flyg varje person åker med.
#Slumpa start- och slutvärde i en lista, skicka in i fitness function. Returnera en lista med bästa lösning.
#Gör denna sökning 1000 gånger per function call och spara fitnessfunktionens resultat för 50 function calls.
#Någon slutsats?