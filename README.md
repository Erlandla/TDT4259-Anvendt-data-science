# TDT4259-Anvendt-data-science

## Data exploration

Se på consumption data for byene, lek litt med det og lag litt grafer etc. God måte å bli kjent med datasettet, og gjør det også lettere å se hva som må gjrøes i preprocessing

## Data preparation
Med andre ord, hva må precocesses i dataen før vi kan bruke ulike modeller ?

- Er det outliers i datasettet som vi må fjerne? z-score er et eksempel på en måte å fjerne outliers
- Data cleaning: Lite sannsynlig at vi har "missing values" i datasettet, men hvis vi fjerner outliers burde vi bruke en metode for å erstatte dem evt.
- Feature engineering: Spesielt relevant med "feature engineering" på tid. F.eks. gjøre om til dag eller uke i steden for hver time.
- Data splitting: Gjør om til training og test sets for modellering.
