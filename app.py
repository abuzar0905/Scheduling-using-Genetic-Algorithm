import streamlit as st
import pandas as pd
import random

# --- 1. GENETIC ALGORITHM LOGIC ---

# --- GA Constants ---
POPULATION_SIZE = 50
NUM_GENERATIONS = 100
# Set to 8 slots as per your last request
NUM_TIME_SLOTS = 8  

# --- GA Function: Create a single schedule (Chromosome) ---
def create_individual(program_list):
    """
    Creates a random schedule (an 'individual') by sampling 8 programs.
    """
    schedule = random.sample(program_list, NUM_TIME_SLOTS)
    return schedule

# --- GA Function: Create the first population ---
def initialize_population(program_list):
    """
    Creates the initial population of random schedules
    """
    population = []
    for _ in range(POPULATION_SIZE):
        population.append(create_individual(program_list))
    return population

# --- GA Function: Score a schedule (Fitness Function) ---
def calculate_fitness(schedule, ratings_dict):
    """
    Calculates the total rating (fitness) for a given schedule
    """
    fitness = 0
    # Use set() to ensure we only count the fitness of unique programs
    for program_id in set(schedule):
        fitness += ratings_dict.get(program_id, 0)
    return fitness

# --- GA Function: Select parents for breeding (Tournament Selection) ---
def selection(population, ratings_dict):
    """
    Selects a good parent using tournament selection
    """
    tournament = random.sample(population, 5)
    best_individual = max(tournament, key=lambda ind: calculate_fitness(ind, ratings_dict))
    return best_individual

# --- GA Function: Create children (Crossover) ---
def crossover(parent1, parent2, crossover_rate):
    """
    Performs single-point crossover with a given rate (co_r).
    """
    if random.random() < crossover_rate:
        # Crossover point is based on 8 time slots
        point = random.randint(1, NUM_TIME_SLOTS - 1)
        
        child1 = parent1[:point] + [gene for gene in parent2 if gene not in parent1[:point]]
        child2 = parent2[:point] + [gene for gene in parent1 if gene not in parent2[:point]]
        
        # Fill in remaining genes if list is too short (due to duplicates)
        all_genes = list(set(parent1 + parent2))
        
        i = 0
        while len(child1) < NUM_TIME_SLOTS and i < len(all_genes):
            if all_genes[i] not in child1:
                child1.append(all_genes[i])
            i += 1
            
        i = 0
        while len(child2) < NUM_TIME_SLOTS and i < len(all_genes):
            if all_genes[i] not in child2:
                child2.append(all_genes[i])
            i += 1

        # Ensure correct length
        return child1[:NUM_TIME_SLOTS], child2[:NUM_TIME_SLOTS]
    else:
        # No crossover, parents pass on
        return parent1, parent2

# --- GA Function: Randomly change a schedule (Mutation) ---
def mutation(individual, mutation_rate):
    """
    Performs swap mutation with a given rate (mut_r).
    """
    if random.random() < mutation_rate:
        # Pick two random positions (indices) to swap
        idx1, idx2 = random.sample(range(NUM_TIME_SLOTS), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual

# --- Main GA Function: This runs the whole process ---
def run_ga(csv_data, co_r, mut_r):
    """
    This is the main function that runs the entire GA process
    and returns the best schedule as a pandas DataFrame.
    """
    
    # Get the list of unique programs
    available_programs = csv_data['ProgramID'].unique().tolist()
    
    # Prepare data dictionary for fitness calculation
    ratings_dict = pd.Series(csv_data.Rating.values, index=csv_data.ProgramID).to_dict()

    # Initialize population
    population = initialize_population(available_programs)
    
    # Evolve for a number of generations
    for _ in range(NUM_GENERATIONS):
        new_population = []
        
        # Elitism: Keep the best individual from the current generation
        best_of_gen = max(population, key=lambda ind: calculate_fitness(ind, ratings_dict))
        new_population.append(best_of_gen)
        
        # Fill the rest of the new population
        while len(new_population) < POPULATION_SIZE:
            parent1 = selection(population, ratings_dict)
            parent2 = selection(population, ratings_dict)
            
            # Pass the user-defined rates to the functions
            child1, child2 = crossover(parent1, parent2, co_r)
            child1 = mutation(child1, mut_r)
            child2 = mutation(child2, mut_r)
            
            new_population.extend([child1, child2])
        
        population = new_population[:POPULATION_SIZE]

    # Get the best schedule from the final population
    best_schedule_ids = max(population, key=lambda ind: calculate_fitness(ind, ratings_dict))
    
    # Format the final schedule into a nice DataFrame
    schedule_details = csv_data.drop_duplicates(subset=['ProgramID'])
    
    final_schedule_df = pd.DataFrame({'ProgramID': best_schedule_ids})
    final_schedule_df = final_schedule_df.merge(schedule_details, on='ProgramID', how='left')
    
    # Removed "Time Slot" column as requested.
    # The final table will show these 4 columns for the 8 selected programs.
    final_schedule_df = final_schedule_df[['ProgramID', 'ProgramName', 'Genre', 'Rating']]
    
    # Calculate the total fitness (rating) of the best schedule
    total_fitness = final_schedule_df['Rating'].sum()

    return final_schedule_df, total_fitness

# -----------------------------------------------------------------
# --- 2. STREAMLIT INTERFACE ---
# -----------------------------------------------------------------

st.title('Genetic Algorithm for TV Scheduling')

st.sidebar.header('GA Parameters Input')

# Crossover Rate (CO_R) slider
co_r = st.sidebar.slider(
    'Crossover Rate (CO_R)',
    min_value=0.0,
    max_value=0.95, # Range 0 to 0.95
    value=0.8,      # Default 0.8
    step=0.05
)

# Mutation Rate (MUT_R) slider
mut_r = st.sidebar.slider(
    'Mutation Rate (MUT_R)',
    min_value=0.01, # Range 0.01 to 0.05
    max_value=0.05, # Range 0.01 to 0.05
    value=0.02,     # Using 0.02 as a logical default *within* the required range
    step=0.01
)

st.sidebar.write('---')

# --- 3. RUN ALGORITHM AND DISPLAY RESULTS ---

if st.sidebar.button('Run Genetic Algorithm'):
    
    try:
        # Load the modified CSV file
        CSV_FILE_NAME = 'program_ratings.csv'
        ratings_data = pd.read_csv(CSV_FILE_NAME)
        
        # Check if CSV has required columns and at least 8 unique programs
        if 'ProgramID' not in ratings_data.columns or 'Rating' not in ratings_data.columns:
            st.error(f"Error: Your CSV file ('{CSV_FILE_NAME}') must contain 'ProgramID' and 'Rating' columns.")
        elif len(ratings_data['ProgramID'].unique()) < NUM_TIME_SLOTS:
            st.error(f"Error: Your CSV file must contain at least {NUM_TIME_SLOTS} unique programs to fill the schedule.")
        else:
            with st.spinner('Evolving schedules... Please wait.'):
                # Run the GA with user-defined parameters
                final_schedule, total_fitness = run_ga(ratings_data, co_r, mut_r)
            
            st.success('Algorithm run complete!')
            
            # Document the parameters used
            st.subheader('Parameters Used for This Trial')
            st.code(f"Crossover Rate: {co_r}\nMutation Rate: {mut_r}")
            
            # Display the resulting schedule in a table
            st.subheader('Resulting Schedule')
            st.write(f"**Total Schedule Rating (Fitness): {total_fitness:.2f}**")
            st.dataframe(final_schedule)

    except FileNotFoundError:
        st.error(f"Error: The CSV file ('{CSV_FILE_NAME}') was not found. Please make sure it's in your repository and named correctly.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.info('Set your parameters in the sidebar and click "Run Genetic Algorithm" to start.')
