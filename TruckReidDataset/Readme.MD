The folder TruckReidDataset consists of truck crops partitioned into the test and training split. Information about the crops is provided in files corresponding to their split. Each split has a pkl file which can be opened directly via [pandas](https://pandas.pydata.org/docs/). The same information is also provided in txt files which should be readable by any csv reader. 

The train split file contains the following columns: 

id,reid,timestamp,filepath,features,direction,drone,position_utm

The "id" is a unique identifier for any vehicle, and any seperate ids that were determined to be the same vehicle were assigned the same "reid" identifier. If an id was not associated with any other id, the "reid" column is blank. 

The "timestamp" provides temporal information for each detection which will be vital in reid identification. 

The "features" column is a numpy array represented as a string that consists of a reid feature vector from a model that was trained on [vehicle data](https://github.com/JDAI-CV/fast-reid/blob/master/configs/VERIWild/bagtricks_R50-ibn.yml). You can choose to use this as part of your solution

The "direction" column describes the direction of travel of the vehicle. This combined with the drone information impose certain constraints on the possible reids:

+ Only vehicles travelling in the same direction can be associated. 
+ The drones are organized as follows South to North: 1 -> 2 -> 3
    * This means, for example:
        - Vehicles that were seen northbound in drone 3 will not be reidentified in drone 1 or 2 
        - Vehicles that were seen southbound in drone 1 will not be reidentified in drone 2 or 3
        - Vehicles that are seen northbound in drone 1 can only be reidentified at times after their appearance in drone 1

The "position_utm" column gives a rough location (in UTM coordinates) for each detection based on its drone this can be used to compute speed which can also help in determining reids. 


The test split is formatted the same, but without a "reid" column this is what you must determine. 
