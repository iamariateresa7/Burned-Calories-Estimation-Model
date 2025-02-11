:- module(kb, [ 
    calories_per_kg/2,
    calories_burned/4,
    bmi/3,
    fitness_level/2,
    workout_type/2,
    recommended_workout/3,
    intensity_threshold/2,
    intensity_category/2,
    recommended_intensity/4,
    suggested_duration/2,
    adjust_duration/3,
    optimal_duration/4
]).

/** <module> Knowledge Base per Calorie e Fitness

    Questo modulo fornisce regole per stimare le calorie bruciate,
    calcolare il BMI, classificare il livello di fitness e
    suggerire allenamenti basati su peso e altezza.
*/

/** calories_per_kg(+Workout_Type, -CaloriesPerKg)
    
    Determina il consumo calorico per kg all'ora in base al tipo di allenamento.
    
    @param Workout_Type Tipo di allenamento (0: Cardio, 1: HIIT, 2: Strength Training, 3: Yoga)
    @param CaloriesPerKg Calorie bruciate per kg per ora
*/
calories_per_kg(0, 8.0).   % Cardio
calories_per_kg(1, 10.0).  % HIIT
calories_per_kg(2, 6.0).   % Strength Training
calories_per_kg(3, 3.5).   % Yoga

/** calories_burned(+Workout_Type, +Duration, +Weight, -Calories)
    
    Stima le calorie bruciate in base al tipo di allenamento, durata e peso corporeo.
    
    @param Workout_Type Tipo di allenamento
    @param Duration Durata dell'allenamento (ore)
    @param Weight Peso corporeo (kg)
    @param Calories Calorie bruciate stimate
*/
calories_burned(Workout_Type, Duration, Weight, Calories) :-
    calories_per_kg(Workout_Type, CalPerKg),
    Calories is CalPerKg * Weight * Duration.

/** bmi(+Weight, +Height, -BMI)
    
    Calcola il BMI (Body Mass Index).
    
    @param Weight Peso corporeo (kg)
    @param Height Altezza (m)
    @param BMI Indice di massa corporea calcolato
*/
bmi(Weight, Height, BMI) :-
    Height > 0,
    BMI is Weight / (Height * Height).

/** fitness_level(+BMI, -Level)
    
    Classifica il livello di fitness in base al BMI.
    
    @param BMI Indice di massa corporea
    @param Level Categoria di fitness (underweight, normal, overweight, obese)
*/
fitness_level(BMI, underweight) :- BMI < 18.5.
fitness_level(BMI, normal) :- BMI >= 18.5, BMI =< 24.9.
fitness_level(BMI, overweight) :- BMI >= 25, BMI =< 29.9.
fitness_level(BMI, obese) :- BMI >= 30.

/** workout_type(+FitnessLevel, -Workout_Type)
    
    Associa il livello di fitness con il tipo di allenamento consigliato.
    
    @param FitnessLevel Livello di fitness (underweight, normal, overweight, obese)
    @param Workout_Type Tipo di allenamento consigliato
*/
workout_type(underweight, 2). % Strength Training
workout_type(normal, 0). % Cardio
workout_type(overweight, 1). % HIIT
workout_type(obese, 3). % Yoga

/** recommended_workout(+Weight, +Height, -Workout_Type)
    
    Determina il tipo di allenamento consigliato in base al peso e all'altezza.
    
    @param Weight Peso corporeo (kg)
    @param Height Altezza (m)
    @param Workout_Type Tipo di allenamento consigliato
*/
recommended_workout(Weight, Height, Workout_Type) :-
    bmi(Weight, Height, BMI),
    fitness_level(BMI, Level),
    workout_type(Level, Workout_Type).

/** intensity_threshold(+Intensità, -Soglia)
 *
 * Definisce le soglie di calorie bruciate per determinare il livello di intensità dell'allenamento.
 *
 * @param Intensità Livello di intensità dell'allenamento (high, moderate)
 * @param Soglia Numero minimo di calorie bruciate per rientrare in tale livello di intensità
 */
intensity_threshold(high, 800).
intensity_threshold(moderate, 400).

/** intensity_category(+Calories, -Intensity)
    
    Classifica l'intensità dell'allenamento in base alle calorie bruciate.
    
    @param Calories Calorie bruciate
    @param Intensity Intensità dell'allenamento (low, moderate, high)
*/
intensity_category(Calories, high) :-
    intensity_threshold(high, Threshold),
    Calories > Threshold.
intensity_category(Calories, moderate) :-
    intensity_threshold(high, HighThreshold),
    intensity_threshold(moderate, ModerateThreshold),
    Calories =< HighThreshold,
    Calories >= ModerateThreshold.
intensity_category(Calories, low) :-
    intensity_threshold(moderate, Threshold),
    Calories < Threshold.

/** recommended_intensity(+Weight, +Height, -Intensity, +Duration)
    
    Determina l'intensità dell'allenamento raccomandato basata su peso, altezza e durata.
    
    @param Weight Peso corporeo (kg)
    @param Height Altezza (m)
    @param Intensity Intensità consigliata (low, moderate, high)
    @param Duration Durata dell'allenamento (ore)
*/
recommended_intensity(Weight, Height, Intensity, Duration) :-
    recommended_workout(Weight, Height, Workout_Type),
    calories_burned(Workout_Type, Duration, Weight, Calories),
    intensity_category(Calories, Intensity).

/** suggested_duration(+Intensità, -Durata)
 *
 * Determina la durata consigliata di un allenamento in base alla sua intensità.
 *
 * @param Intensità Livello di intensità dell'allenamento (high, moderate, low)
 * @param Durata Durata consigliata dell'allenamento in ore
 */
suggested_duration(high, 0).
suggested_duration(moderate, 0.5).
suggested_duration(low, 1).

/** adjust_duration(+Intensity, +BaseDuration, -FinalDuration)
    
    Regola la durata dell'allenamento in base all'intensità.
    
    @param Intensity Intensità dell'allenamento (low, moderate, high)
    @param BaseDuration Durata iniziale (ore)
    @param FinalDuration Durata finale consigliata (ore)
*/
adjust_duration(Intensity, BaseDuration, FinalDuration) :-
    suggested_duration(Intensity, SuggestedDuration),
    FinalDuration is BaseDuration + SuggestedDuration.

/** optimal_duration(+Weight, +Height, +Duration, -OptimalDuration)
    
    Calcola la durata ottimale dell'allenamento basata su intensità e durata iniziale.
    
    @param Weight Peso corporeo (kg)
    @param Height Altezza (m)
    @param Duration Durata iniziale dell'allenamento (ore)
    @param OptimalDuration Durata ottimale consigliata (ore)
*/
optimal_duration(Weight, Height, Duration, OptimalDuration) :-
    recommended_intensity(Weight, Height, Intensity, Duration),
    adjust_duration(Intensity, Duration, OptimalDuration).