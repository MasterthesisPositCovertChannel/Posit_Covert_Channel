using Random
using Statistics
using JSON
using CSV
using DataFrames
using SoftPosit
using StatsBase
using Base.Threads

# ---------------------------------------------
# Utility Functions
# ---------------------------------------------

"""
    NaR16()

Returns a 16-bit Posit value representing "Not a Real" (NaR).
NaR (sign bit 1, rest 0) is a special Posit value used to represent invalid results (similar to NaN in IEEE754).
"""
NaR16() = reinterpret(Posit16, UInt16(0x8000))

"""
    NaR32()

Returns a 32-bit Posit value representing "Not a Real" (NaR).
"""
NaR32() = reinterpret(Posit32, UInt32(0x80000000))

"""
    load_config(config_file::String) -> Dict

Loads the JSON configuration file and return it as a dictionary.
"""
function load_config(config_file)
    config = JSON.parsefile(config_file)
    return config
end

"""
    load_mapping_intervals(bit_size::Int) -> (DataFrame, DataFrame)

Loads both positive and negative interval mappings for the given bit_size.
We assume files:
- mapping_intervals_16_positive.csv / mapping_intervals_16_negative.csv
- mapping_intervals_32_positive.csv / mapping_intervals_32_negative.csv

Returns a tuple (intervals_df_positive, intervals_df_negative).

"""
function load_mapping_intervals(bit_size::Int)
    pos_file = "mapping_intervals_$(bit_size)_positive.csv"
    neg_file = "mapping_intervals_$(bit_size)_negative.csv"

    intervals_df_positive = CSV.read(
        pos_file,
        DataFrame;
        missingstring=["-"],
        types=Dict(
            "f_lower" => Union{Missing, Float64},
            "f_upper" => Union{Missing, Float64},
            "num_floats_in_interval" => Union{Missing, Int},
            "num_posits_in_interval" => Union{Missing, Int}
        )
    )

    intervals_df_negative = CSV.read(
        neg_file,
        DataFrame;
        missingstring=["-"],
        types=Dict(
            "f_lower" => Union{Missing, Float64},
            "f_upper" => Union{Missing, Float64},
            "num_floats_in_interval" => Union{Missing, Int},
            "num_posits_in_interval" => Union{Missing, Int}
        )
    )

    return intervals_df_positive, intervals_df_negative
end

"""
    isNaR(p, bit_size::Int) -> Bool

Checks if a given posit value p (Posit16 or Posit32) is NaR.
"""
function isNaR(p, bit_size::Int)
    if bit_size == 16
        p_bits = reinterpret(UInt16, p)
        return p_bits == UInt16(0x8000)
    else
        p_bits = reinterpret(UInt32, p)
        return p_bits == UInt32(0x80000000)
    end
end

"""
    find_interval_index(value::Float64, intervals_df_positive::DataFrame, intervals_df_negative::DataFrame)

Find which interval the given value falls into. If value >= 0, use positive intervals.
If value < 0, use negative intervals.

"""
function find_interval_index(value::Float64, intervals_df_positive::DataFrame, intervals_df_negative::DataFrame)
    df = value >= 0 ? intervals_df_positive : intervals_df_negative

    if value >= 0
        # Search in positive intervals
        for i in 1:nrow(df)
            f_lower = df.f_lower[i]
            f_upper = df.f_upper[i]
            if value >= f_lower && value <= f_upper
                return i, :positive
            end
        end
    else
        # Search in negative intervals
        # Negative intervals should have f_lower < f_upper but both negative
        for i in 1:nrow(df)
            f_lower = df.f_lower[i]
            f_upper = df.f_upper[i]
            # Both should be negative or zero. Check if value lies within them
            if value >= f_lower && value <= f_upper
                return i, :negative
            end
        end
    end

    return nothing, value >= 0 ? :positive : :negative
end

"""
    float_to_posit(x, bit_size::Int)

Converts a floating-point number x into a Posit representation with the given bit size.
"""
function float_to_posit(x, bit_size::Int)
    if bit_size == 16
        return Posit16(x)
    else
        return Posit32(x)
    end
end

"""
    posit_to_float(p, bit_size::Int)

Converts a posit value p back to a IEEE754 floating-point number (Float16 or Float32).
"""
function posit_to_float(p, bit_size::Int)
    if bit_size == 16
        return Float16(p)
    else
        return Float32(p)
    end
end

# ---------------------------------------------
# Sensor Data Loading Function
# ---------------------------------------------

"""
    load_sensor_data(sensor_data_folder::String) -> Vector{Float64}

Loads real-world gas sensor data from given CSV files, to be used as overt traffic.
"""
function load_sensor_data(sensor_data_folder::String)::Vector{Float64}
    # Get list of all files in the sensor data folder
    files = readdir(sensor_data_folder)
    sensor_data = Float64[]
    for file in files
        if endswith(file, ".csv")
            file_path = joinpath(sensor_data_folder, file)
            println("Loading sensor data from file: $file_path")
            # Read the CSV file into a DataFrame
            df = CSV.read(file_path, DataFrame; missingstring=["-"])
            sensor_columns = ["R1 (MOhm)", "R2 (MOhm)", "R3 (MOhm)", "R4 (MOhm)", "R5 (MOhm)", "R6 (MOhm)", "R7 (MOhm)", "R8 (MOhm)", "R9 (MOhm)", "R10 (MOhm)", "R11 (MOhm)", "R12 (MOhm)", "R13 (MOhm)", "R14 (MOhm)"]
            missing_cols = setdiff(sensor_columns, names(df))
            if !isempty(missing_cols)
                println("Missing columns in file $file_path: $missing_cols")
                continue
            end
            # Append all valid sensor readings
            for col_name in sensor_columns
                readings = df[!, col_name]
                valid_readings = [Float64(r) for r in readings if !ismissing(r)]
                append!(sensor_data, valid_readings)
            end
        else
            println("Skipping file: $file")
        end
    end
    return sensor_data
end

# ---------------------------------------------
# Overt Message Functions
# ---------------------------------------------

"""
    generate_random_overt_message(config, bit_size::Int) -> Vector{Float64}

Wrapper that generates the overt message based on the given configuration when use_sensor_data = false.
Prepares the necessary parameters and then calls generate_overt_message.
Returns a vector of decimal values as Float64 for storage convenience.
"""
function generate_random_overt_message(config, bit_size::Int)
    overt_length = config["overt_message_length"]
    decimal_range = (Float64(config["decimal_range"]["min"]), Float64(config["decimal_range"]["max"]))
    inject_nans = config["inject_nans"]
    nan_injection_ratio = config["nan_injection_ratio"]
    inf_injection_ratio = config["inf_injection_ratio"]
    overt_message_decimal, _ = generate_overt_message(
        bit_size, overt_length, decimal_range, inject_nans, nan_injection_ratio, inf_injection_ratio
    )
    return overt_message_decimal
end

"""
    generate_overt_message(...)

Creates the overt message.
Selects random values in the given range, injects NaNs/Infs if enabled.
Values are stored as Float64 but may represent Float16/Float32 precision.
"""
function generate_overt_message(bit_size::Int, length::Int, decimal_range::Tuple{Float64, Float64}, inject_nans::Bool, nan_injection_ratio::Float64, inf_injection_ratio::Float64)
    min_val, max_val = decimal_range
    overt_message_decimal = Vector{Float64}(undef, length)
    overt_message_binary = Vector{Int}(undef, length)
    for i in 1:length
        rand_val = rand()
        if inject_nans && rand_val < nan_injection_ratio
            num_float = generate_random_nan(bit_size)
        elseif inject_nans && rand_val < nan_injection_ratio + inf_injection_ratio
            # Inject an Inf (positive or negative)
            if bit_size == 16
                num_float = rand() < 0.5 ? Float16(Inf) : Float16(-Inf)
            else
                num_float = rand() < 0.5 ? Float32(Inf) : Float32(-Inf)
            end
        else
            num = rand() * (max_val - min_val) + min_val
            # Convert to chosen precision for actual covert traffic simulation
            num_float = (bit_size == 16) ? Float16(num) : Float32(num)
        end

        # Store decimal as Float64 for file writing convenience
        overt_message_decimal[i] = Float64(num_float)

        # Store the binary representation
        bin_str = bit_size == 16 ? bitstring(Float16(num_float)) : bitstring(Float32(num_float))
        overt_message_binary[i] = parse(Int, bin_str; base=2)
    end
    return overt_message_decimal, overt_message_binary
end

"""
    generate_random_nan(bit_size::Int)

Generates a random IEEE754 NaN for the given bit_size (16 or 32).
"""
function generate_random_nan(bit_size::Int)
    if bit_size == 16
        # For Float16, create a NaN by setting exponent all ones and a non-zero fraction
        exponent_bits = UInt16(0b11111)
        fraction_bits = UInt16(rand(0b00001:0b11111))
        sign_bit = UInt16(rand(0:1))
        bits = (sign_bit << 15) | (exponent_bits << 10) | fraction_bits
        return reinterpret(Float16, bits)
    else
        # For Float32, similarly set exponent all ones and non-zero fraction
        exponent_bits = UInt32(0b11111111)
        fraction_bits = UInt32(rand(0x000001:0x7FFFFF)) # Avoid zero fraction
        sign_bit = UInt32(rand(0:1))
        bits = (sign_bit << 31) | (exponent_bits << 23) | fraction_bits
        return reinterpret(Float32, bits)
    end
end

"""
    write_overt_message(overt_message_decimal, bit_size, file_paths)

Writes the overt message to decimal and binary files.
Decimal values are converted to chosen precision (Float16/Float32) before writing,
to simulate the actual precision.
"""
function write_overt_message(overt_message_decimal::Vector{<:Real}, bit_size::Int, file_paths::Dict{String, String})
    # Write decimal values at chosen precision
    open(file_paths["overt_decimal"], "w") do io
        for num in overt_message_decimal
            if bit_size == 16
                write(io, string(Float16(num)), "\n")
            else
                write(io, string(Float32(num)), "\n")
            end
        end
    end

    # Write binary representations
    open(file_paths["overt_binary"], "w") do io
        for num in overt_message_decimal
            # Convert back to chosen format before bitstring
            bin_str = (bit_size == 16) ? bitstring(Float16(num)) : bitstring(Float32(num))
            write(io, "$bin_str\n")
        end
    end
end

"""
    read_overt_message_binary(file_path::String) -> Vector{String}

Reads the binary lines from a file. Each line represents a float in binary form.
"""
function read_overt_message_binary(file_path::String)::Vector{String}
    return readlines(file_path)
end

"""
    read_overt_message_decimal(file_path::String, bit_size::Int)

Reads the decimal overt message values and convert them to chosen format (Float16/Float32).
"""
function read_overt_message_decimal(file_path::String, bit_size::Int)::Vector{Union{Float16, Float32}}
    overt_message_decimal = Union{Float16, Float32}[]
    open(file_path, "r") do io
        for line in eachline(io)
            val_64 = parse(Float64, strip(line))
            val_converted = bit_size == 16 ? Float16(val_64) : Float32(val_64)
            push!(overt_message_decimal, val_converted)
        end
    end
    return overt_message_decimal
end

# ---------------------------------------------
# Covert Message Functions
# ---------------------------------------------

"""
    estimate_capacity(file_paths, intervals_df, bit_size) -> Int

Estimates how many bits in total can be embedded in the given overt traffic, based on
the precomputed intervals and how many Posits values exist per float in the interval.
"""
function estimate_capacity(file_paths::Dict{String, String}, intervals_df_positive::DataFrame, intervals_df_negative::DataFrame, bit_size::Int)
    overt_message_decimal = read_overt_message_decimal(file_paths["overt_decimal"], bit_size)
    total_capacity_bits = 0
    for num in overt_message_decimal
        num_float64 = Float64(num)
        interval_idx, mode = find_interval_index(num_float64, intervals_df_positive, intervals_df_negative)
        if interval_idx !== nothing
            df = (mode == :positive) ? intervals_df_positive : intervals_df_negative
            num_floats = df.num_floats_in_interval[interval_idx]
            num_posits = df.num_posits_in_interval[interval_idx]
            ratio = num_posits / num_floats
            if ratio > 1
                total_capacity_bits += floor(Int, log2(ratio))
            end
        end
    end
    return total_capacity_bits
end

"""
    generate_or_load_covert_message(config, total_capacity_bits)

Generates a random covert message or loads a premade one.
If auto_covert_length = true, uses the maximum capacity.
If a specified length is bigger than capacity, truncates it.
"""
function generate_or_load_covert_message(config::Dict, total_capacity_bits::Int)
    if config["use_premade_covert"]
        covert_file_path = config["premade_covert_path"]
        covert_is_text = config["premade_covert_is_text"]
        covert_message = load_premade_covert_message(covert_file_path, covert_is_text)
        actual_covert_length = length(covert_message)
    else
        if config["auto_covert_length"]
            println("Auto Covert Length is enabled.")
            println("Generating covert message with maximum capacity: $total_capacity_bits bits.")
            covert_message = generate_covert_message(total_capacity_bits)
            actual_covert_length = total_capacity_bits
        else
            covert_length = config["covert_message_length"]
            if covert_length > total_capacity_bits
                println("Warning: Specified covert message length exceeds capacity.")
                println("Truncating covert message length to maximum capacity: $total_capacity_bits bits.")
                covert_length = total_capacity_bits
            end
            println("Generating covert message with specified length: $covert_length bits.")
            covert_message = generate_covert_message(covert_length)
            actual_covert_length = covert_length
        end
    end
    return covert_message, actual_covert_length
end

"""
    generate_covert_message(length::Int) -> Vector{Int}

Generates a covert message as a random sequence of bits.
"""
generate_covert_message(length::Int) = rand(0:1, length)

"""
    load_premade_covert_message(file_path::String, is_text::Bool) -> Vector{Int}

Loads a premade covert message from a file. If is_text = true, converts characters to their
8-bit binary representation. Otherwise, treats each line as bits.
"""
function load_premade_covert_message(file_path::String, is_text::Bool)::Vector{Int}
    covert_message = Int[]
    open(file_path, "r") do io
        for line in eachline(io)
            if is_text
                for char in line
                    char_code = UInt8(codepoint(char))
                    char_bits_str = bitstring(char_code)
                    char_bits = [parse(Int, ch) for ch in collect(char_bits_str)]
                    append!(covert_message, char_bits)
                end
            else
                bits_str = strip(line)
                bits_list = [parse(Int, ch) for ch in bits_str]
                append!(covert_message, bits_list)
            end
        end
    end
    return covert_message
end

"""
    write_covert_message(covert_message, file_paths)

Writes the covert message (bits) to a file as a sequence of '0' and '1'.
"""
function write_covert_message(covert_message::Vector{Int}, file_paths::Dict{String, String})
    open(file_paths["covert_message"], "w") do io
        for bit in covert_message
            write(io, "$bit")
        end
    end
end

# ---------------------------------------------
# Embedding and Extraction Functions
# ---------------------------------------------

"""
    embed_covert_data(...)

Performs the actual embedding of covert bits into the Posit LSBs.
- Converts each overt IEEE754 float to Posit
- Determines how many bits can be embedded based on intervals
- Embeds covert bits if possible
"""
function embed_covert_data(overt_message_binary::Vector{String}, covert_message::Vector{Int}, bit_size::Int, intervals_df_positive::DataFrame, intervals_df_negative::DataFrame, nan_handling::String)
    embedded_posits = Vector{Any}(undef, length(overt_message_binary))
    covert_index = 1
    total_bits = length(covert_message)
    total_bits_embedded = 0  # Counter for total bits actually embedded

    @threads for idx in 1:length(overt_message_binary)
        bin_str = overt_message_binary[idx]
        # Parse binary line back to float (to decide how to embed)
        if bit_size == 16
            p_bits = parse(UInt16, bin_str; base=2)
            num = Float64(reinterpret(Float16, p_bits))
        else
            p_bits = parse(UInt32, bin_str; base=2)
            num = Float64(reinterpret(Float32, p_bits))
        end

        num_float64 = Float64(num)

        # Handle NaN/Inf according to chosen NaN-Handling strategy
        if isnan(num_float64) || isinf(num_float64)
            if nan_handling == "convert_to_NaR"
                p = (bit_size == 16) ? NaR16() : NaR32()
                embedded_posits[idx] = p
            elseif nan_handling == "ignore"
                embedded_posits[idx] = num  # Just keep original value
            else
                error("Invalid option for handling NaNs and infinities.")
            end
            continue
        end

        # Find interval and determine how many bits are available
        interval_idx, mode = find_interval_index(num_float64, intervals_df_positive, intervals_df_negative)
        if interval_idx === nothing
            # No embedding possible here
            embedded_posits[idx] = float_to_posit(num, bit_size)
            continue
        end

        df = (mode == :positive) ? intervals_df_positive : intervals_df_negative
        num_floats = df.num_floats_in_interval[interval_idx]
        num_posits = df.num_posits_in_interval[interval_idx]
        ratio = num_posits / num_floats
        available_bits = floor(Int, log2(ratio))

        # If no bits available or no covert bits left to embed, just store as Posit
        if available_bits <= 0 || covert_index > total_bits
            p = float_to_posit(num, bit_size)
            embedded_posits[idx] = p
            continue
        end

        # Convert float to Posit before embedding
        p = float_to_posit(num, bit_size)

        # Get posit bits as integer
        if bit_size == 16
            p_bits = reinterpret(UInt16, p)
            mask = ~(UInt16(2^available_bits - 1))
        else
            p_bits = reinterpret(UInt32, p)
            mask = ~(UInt32(2^available_bits - 1))
        end

        # Clear LSBs to embed covert bits
        p_bits &= mask

        # Extract the covert bits we can embed here
        bits_to_embed = covert_message[covert_index:min(covert_index + available_bits - 1, total_bits)]
        num_bits_to_embed = length(bits_to_embed)

        # Place covert bits into LSBs
        for i in 1:num_bits_to_embed
            if bit_size == 16
                p_bits |= (UInt16(bits_to_embed[i]) << (i - 1))
            else
                p_bits |= (UInt32(bits_to_embed[i]) << (i - 1))
            end
        end

        # Reinterpret modified bits as a Posit
        p_embedded = (bit_size == 16) ? reinterpret(Posit16, p_bits) : reinterpret(Posit32, p_bits)
        embedded_posits[idx] = p_embedded
		covert_index += num_bits_to_embed
		total_bits_embedded += num_bits_to_embed
    end

    println("Total bits embedded: $total_bits_embedded")
    return embedded_posits, total_bits_embedded
end

"""
    write_modified_message(embedded_posits, bit_size, file_paths)

Writes the Posit-modified embedded message in binary form to a file.
"""
function write_modified_message(embedded_posits::Vector{Any}, bit_size::Int, file_paths::Dict{String, String})
    open(file_paths["modified_binary"], "w") do io
        for p in embedded_posits
            p_bits = if typeof(p) <: SoftPosit.AbstractPosit
                (bit_size == 16) ? bitstring(reinterpret(UInt16, p)) : bitstring(reinterpret(UInt32, p))
            else
                # p is a float (NaN/Inf case if ignored)
                (bit_size == 16) ? bitstring(Float16(p)) : bitstring(Float32(p))
            end
            write(io, "$p_bits\n")
        end
    end
end

"""
    extract_covert_data(...)

Reads the LSBs of Posits based on interval calculations and reconstructs the covert bits.
"""
function extract_covert_data(embedded_posits_loaded, bit_size::Int, intervals_df_positive::DataFrame, intervals_df_negative::DataFrame, total_bits_embedded, nan_handling::String)::Vector{String}
    extracted_bits = String[]
    @threads for idx in 1:length(embedded_posits_loaded)
        p = embedded_posits_loaded[idx]

        # Convert Posit back to float
        float_value = (bit_size == 16) ? reinterpret(Float16, reinterpret(UInt16, p)) :
                                         reinterpret(Float32, reinterpret(UInt32, p))

        # Skip if NaN/Inf as no covert data is embedded there
        if isnan(float_value) || isinf(float_value)
            continue
        end

        # Find interval to determine how many bits might have been embedded
        interval_idx, mode = find_interval_index(Float64(posit_to_float(p, bit_size)), intervals_df_positive, intervals_df_negative)
        if interval_idx === nothing
            continue
        end

        df = (mode == :positive) ? intervals_df_positive : intervals_df_negative
        num_floats = df.num_floats_in_interval[interval_idx]
        num_posits = df.num_posits_in_interval[interval_idx]
        ratio = num_posits / num_floats
        if ratio <= 1
            continue
        end

        available_bits = floor(Int, log2(ratio))
        if typeof(p) <: SoftPosit.AbstractPosit
            p_bits = (bit_size == 16) ? reinterpret(UInt16, p) : reinterpret(UInt32, p)
            # Extract LSBs
            for i in 1:available_bits
                push!(extracted_bits, string((p_bits >> (i - 1)) & 1))
            end
        end
        if length(extracted_bits) >= total_bits_embedded
            break
        end
    end
    println("Total bits extracted: $(length(extracted_bits))")
    return extracted_bits
end

"""
    write_extracted_covert_message(extracted_covert_data, file_paths, config)

Writes the extracted covert bits to a file.
"""
function write_extracted_covert_message(extracted_covert_data::Vector{String}, file_paths::Dict{String, String}, config::Dict)
    open(file_paths["extracted_covert"], "w") do io
        write(io, join(extracted_covert_data, ""))
    end
end

"""
    interpret_modified_message_as_float(modified_message_binary, bit_size)

If enabled, interprets the Posit-modified message as IEEE floats for inspection.
Writes the result into file.
"""
function interpret_modified_message_as_float(modified_message_binary::Vector{String}, file_paths::Dict{String, String}, bit_size::Int)
    modified_message_floats = Vector{Union{Float16, Float32}}(undef, length(modified_message_binary))
    for i in 1:length(modified_message_binary)
        bin_str = modified_message_binary[i]
        val_64 = if bit_size == 16
            Float64(reinterpret(Float16, parse(UInt16, bin_str; base=2)))
        else
            Float64(reinterpret(Float32, parse(UInt32, bin_str; base=2)))
        end

        # Convert to chosen format before writing
        modified_message_floats[i] = (bit_size == 16) ? Float16(val_64) : Float32(val_64)
    end

    open(file_paths["modified_decimal_as_float"], "w") do io
        for num in modified_message_floats
            if bit_size == 16
                write(io, string(Float16(num)), "\n")
            else
                write(io, string(Float32(num)), "\n")
            end
        end
    end

    return modified_message_floats
end

# ---------------------------------------------
# Reconstruction Functions
# ---------------------------------------------

"""
    reconstruct_overt_message(embedded_posits, bit_size, nan_handling)

Reconstructs the overt message from the embedded posits.
If NaR is encountered, convert it to a random NaN, if that was the chosen NaN-Handling strategy.
"""
function reconstruct_overt_message(embedded_posits::Vector{Any}, bit_size::Int, nan_handling::String)
    reconstructed_overt_message_decimal = Vector{Float64}(undef, length(embedded_posits))
    @threads for idx in 1:length(embedded_posits)
        p = embedded_posits[idx]

        # Convert back to float
        float_value = if bit_size == 16
            reinterpret(Float16, reinterpret(UInt16, p))
        else
            reinterpret(Float32, reinterpret(UInt32, p))
        end

        if isnan(float_value) || isinf(float_value)
            # Handle exceptional values
            if nan_handling == "ignore"
                reconstructed_overt_message_decimal[idx] = float_value
            else
                # convert_to_NaR: treat NaR as random NaN
                reconstructed_overt_message_decimal[idx] = Float64(posit_to_float(p, bit_size))
            end
        else
            # If p is a Posit
            if typeof(p) <: SoftPosit.AbstractPosit
                if isNaR(p, bit_size)
                    # NaR encountered, generate random NaN
                    reconstructed_overt_message_decimal[idx] = Float64(generate_random_nan(bit_size))
                else
                    # Convert Posit to float (IEEE)
                    converted_float = posit_to_float(p, bit_size)

                    # Check if negative and adjust by one representable unit if necessary
                    if converted_float < 0
                        if bit_size == 16
                            int_val = reinterpret(UInt16, Float16(converted_float))
                            converted_float = Float64(reinterpret(Float16, int_val))
                        else
                            int_val = reinterpret(UInt32, Float32(converted_float))
                            int_val = int_val + UInt32(1) # increment binary representation by 1
                            converted_float = Float64(reinterpret(Float32, int_val))
                        end
                    end

                    reconstructed_overt_message_decimal[idx] = converted_float
                end
            else
                # p is not a Posit (NaN/Inf ignored scenario)
                converted_float = posit_to_float(p, bit_size)

                if converted_float < 0
                    if bit_size == 16
                        int_val = reinterpret(UInt16, Float16(converted_float))
                        int_val = int_val + UInt16(1)
                        converted_float = Float64(reinterpret(Float16, int_val))
                    else
                        int_val = reinterpret(UInt32, Float32(converted_float))
                        int_val = int_val + UInt32(1)
                        converted_float = Float64(reinterpret(Float32, int_val))
                    end
                end

                reconstructed_overt_message_decimal[idx] = converted_float
            end
        end
    end
    return reconstructed_overt_message_decimal
end

"""
    write_reconstructed_overt_message(reconstructed_overt_message_decimal, bit_size, file_paths)

Writes reconstructed overt message to decimal and binary files, converting back to chosen precision.
"""
function write_reconstructed_overt_message(reconstructed_overt_message_decimal::Vector{<:Real}, bit_size::Int, file_paths::Dict{String, String})
    # Decimal
    open(file_paths["reconstructed_decimal"], "w") do io
        for num in reconstructed_overt_message_decimal
            if bit_size == 16
                write(io, string(Float16(num)), "\n")
            else
                write(io, string(Float32(num)), "\n")
            end
        end
    end

    # Binary
    open(file_paths["reconstructed_binary"], "w") do io
        for num in reconstructed_overt_message_decimal
            bin_str = (bit_size == 16) ? bitstring(Float16(num)) : bitstring(Float32(num))
            write(io, "$bin_str\n")
        end
    end
end

# ---------------------------------------------
# Analysis Functions
# ---------------------------------------------

"""
    perform_analysis(...)

Performs analysis on the overt, modified, and reconstructed messages:
- Calculate percentage matches of bits
- Calculate bit-level entropy
- Calculate byte-level entropy

Returns all metrics for logging.
"""
function perform_analysis(overt_message_binary::Vector{String},
                          reconstructed_overt_binary::Vector{String},
                          covert_message::Vector{Int},
                          extracted_covert_data::Vector{String},
                          modified_message_binary::Vector{String},
                          bit_size::Int)

    overt_message_binary_int = parse.(Int, overt_message_binary; base=2)
    reconstructed_overt_binary_int = parse.(Int, reconstructed_overt_binary; base=2)
    extracted_covert_data_int = parse.(Int, extracted_covert_data; base=2)

    # Ensure extracted_covert_data_int has the same length as covert_message
    if length(extracted_covert_data_int) > length(covert_message)
        extracted_covert_data_int = extracted_covert_data_int[1:length(covert_message)]
    elseif length(extracted_covert_data_int) < length(covert_message)
        extracted_covert_data_int = vcat(extracted_covert_data_int, zeros(Int, length(covert_message) - length(extracted_covert_data_int)))
    end

    # Calculate percentage matches
    percentage_match_overt = calculate_percentage_match(overt_message_binary_int, reconstructed_overt_binary_int)
    percentage_match_covert = calculate_percentage_match(covert_message, extracted_covert_data_int)

    # Calculate bit-level entropy
    entropy_overt = calculate_entropy(overt_message_binary)
    entropy_modified = calculate_entropy(modified_message_binary)
    entropy_reconstructed = calculate_entropy(reconstructed_overt_binary)

    # Calculate byte-level entropy
    overt_byte_entropy = calculate_byte_entropy(overt_message_binary, bit_size)
    modified_byte_entropy = calculate_byte_entropy(modified_message_binary, bit_size)
    reconstructed_byte_entropy = calculate_byte_entropy(reconstructed_overt_binary, bit_size)

    return percentage_match_overt,
           percentage_match_covert,
           entropy_overt,
           entropy_modified,
           entropy_reconstructed,
           overt_byte_entropy,
           modified_byte_entropy,
           reconstructed_byte_entropy
end

"""
    calculate_percentage_match(original_bits, reconstructed_bits) -> Float64

Calculates the percentage match between two equal-length bit arrays.
This checks how many bits remain unchanged after processing.
"""
function calculate_percentage_match(original_bits::Vector{Int}, reconstructed_bits::Vector{Int})
    total_bits = length(original_bits)
    matching_bits = count(==(1), map(==(1), original_bits .== reconstructed_bits))
    return (matching_bits / total_bits) * 100
end

"""
    calculate_entropy(message_bits::Vector{Int}) -> Float64

Calculates bit-level entropy by considering every bit (0 or 1) as a symbol.
Splits each integer line into individual bits (as chars '0'/'1'), counts their frequency,
and computes Shannon entropy.
"""
function calculate_entropy(message_bits::Vector{String})
    bits_list = []
    for bits_str in message_bits
        append!(bits_list, collect(bits_str))
    end
    counts = countmap(bits_list)
    total = length(bits_list)
    probabilities = [counts[bit] / total for bit in keys(counts)]
    return -sum(p * log2(p) for p in probabilities)
end

"""
    calculate_byte_entropy(binary_lines::Vector{String}, bit_size::Int) -> Float64

Calculates the byte-level entropy by grouping each line's bits into bytes (8 bits), treating each byte as a symbol.
"""
function calculate_byte_entropy(binary_lines::Vector{String}, bit_size::Int)
    bytes_per_line = div(bit_size, 8)
    if bytes_per_line * 8 != bit_size
        error("bit_size must be a multiple of 8 for byte-level entropy calculation.")
    end

    byte_values = UInt8[]
    for line in binary_lines
        # Extract bytes from the binary string
        for i in 1:bytes_per_line
            start_index = (i - 1) * 8 + 1
            end_index = i * 8
            byte_str = line[start_index:end_index]
            byte_val = parse(UInt8, byte_str; base=2)
            push!(byte_values, byte_val)
        end
    end

    counts = countmap(byte_values)
    total = length(byte_values)
    probabilities = [counts[b] / total for b in keys(counts)]
    entropy = -sum(p * log2(p) for p in probabilities)

    return entropy
end

"""
    write_log(...)

Writes the analysis results and configuration details to a log file.
"""
function write_log(
    bit_size::Int,
    overt_message_length::Int,
    actual_covert_length::Int,
    config::Dict,
    percentage_match_overt::Float64,
    percentage_match_covert::Float64,
    entropy_overt::Float64,
    entropy_modified::Float64,
    entropy_reconstructed::Float64,
    overt_byte_entropy::Float64,
    modified_byte_entropy::Float64,
    reconstructed_byte_entropy::Float64,
    file_paths::Dict{String, String}
)
    open(file_paths["log"], "w") do io
        write(io, "Covert Channel Simulation Log\n")
        write(io, "-----------------------------\n")
        write(io, "Bit Size: $bit_size bit\n")
        write(io, "Overt Message Length: $overt_message_length floats\n")
        write(io, "Covert Message Length: $actual_covert_length bits\n")
        write(io, "Decimal Range: $(config["decimal_range"])\n")
        write(io, "NaN Handling Option: $(config["nan_handling"])\n")
        write(io, "Auto Covert Length: $(config["auto_covert_length"])\n")
        write(io, "Interpret Modified Message as IEEE Floats: $(config["interpret_modified_as_float"])\n")
        write(io, "Percentage Match (Overt to Reconstructed): $percentage_match_overt%\n")
        write(io, "Percentage Match (Covert to Extracted): $percentage_match_covert%\n")
        write(io, "Entropy of Overt Message (Binary): $entropy_overt\n")
        write(io, "Entropy of Modified Message (Binary): $entropy_modified\n")
        write(io, "Entropy of Reconstructed Overt Message (Binary): $entropy_reconstructed\n")
        write(io, "Byte-Level Entropy of Overt Message: $overt_byte_entropy\n")
        write(io, "Byte-Level Entropy of Modified Message: $modified_byte_entropy\n")
        write(io, "Byte-Level Entropy of Reconstructed Overt Message: $reconstructed_byte_entropy\n")
    end
end

# ---------------------------------------------
# Main Function
# ---------------------------------------------

"""
    main()

Main function that calls all other functions:
- Loads configuration
- Generates or loads overt data
- Embeds covert data
- Extracts covert data
- Reconstructs overt data
- Performs analysis and writes results to log file
"""
function main()
    # Load configuration
    config = load_config("config.json")

    # Convert file_paths to a Dictionary of strings
    file_paths_any = config["file_paths"]
    file_paths = Dict{String, String}()
    for (k, v) in file_paths_any
        if isa(v, String)
            file_paths[k] = v
        else
            error("Value for key '$k' in 'file_paths' must be a string.")
        end
    end

    bit_size = config["bit_size"]
	if bit_size != 16 && bit_size != 32
        error("Unsupported bit size for posit conversion.")
    end

    nan_handling = config["nan_handling"]
    inject_nans = config["inject_nans"]
    nan_injection_ratio = config["nan_injection_ratio"]
    inf_injection_ratio = config["inf_injection_ratio"]
    interpret_modified_as_float = config["interpret_modified_as_float"]

    # Load mapping intervals
    intervals_df_positive, intervals_df_negative = load_mapping_intervals(bit_size)

    # Step 1: Overt Message Generation
    if config["use_sensor_data"]
        sensor_data_folder = config["sensor_data_folder"]
        overt_message_decimal = load_sensor_data(sensor_data_folder)
        # If max_overt_length specified, truncate
        if haskey(config, "max_overt_length")
            max_overt_length = config["max_overt_length"]
            if length(overt_message_decimal) > max_overt_length
                overt_message_decimal = overt_message_decimal[1:max_overt_length]
            end
        end
    else
        overt_message_decimal = generate_random_overt_message(config, bit_size)
    end
    overt_message_length = length(overt_message_decimal)
    write_overt_message(overt_message_decimal, bit_size, file_paths)

    # Step 2: Covert Message Generation or Loading
    total_capacity_bits = estimate_capacity(file_paths, intervals_df_positive, intervals_df_negative, bit_size)
    covert_message, actual_covert_length = generate_or_load_covert_message(config, total_capacity_bits)
    write_covert_message(covert_message, file_paths)

    # Step 3: Embedding Covert Message
    overt_message_binary = read_overt_message_binary(file_paths["overt_binary"])
    overt_message_decimal = read_overt_message_decimal(file_paths["overt_decimal"], bit_size)
    embedded_posits, total_bits_embedded = embed_covert_data(overt_message_binary, covert_message, bit_size, intervals_df_positive, intervals_df_negative, nan_handling)
    write_modified_message(embedded_posits, bit_size, file_paths)

    # Step 4: Extract Covert Message
    modified_message_binary = readlines(file_paths["modified_binary"])
    embedded_posits_loaded = Vector{Any}(undef, length(modified_message_binary))
    for i in 1:length(modified_message_binary)
        bin_str = modified_message_binary[i]
        if bit_size == 16
            p_bits = parse(UInt16, bin_str; base=2)
            embedded_posits_loaded[i] = reinterpret(Posit16, p_bits)
        else
            p_bits = parse(UInt32, bin_str; base=2)
            embedded_posits_loaded[i] = reinterpret(Posit32, p_bits)
        end
    end
    extracted_covert_data = extract_covert_data(embedded_posits_loaded, bit_size, intervals_df_positive, intervals_df_negative, total_bits_embedded, nan_handling)
    write_extracted_covert_message(extracted_covert_data, file_paths, config)

    # Optionally, interpret modified message as IEEE floats
    if interpret_modified_as_float
        _ = interpret_modified_message_as_float(modified_message_binary, file_paths, bit_size)
    end

    # Step 5: Reconstruct Overt Message
    reconstructed_overt_message_decimal = reconstruct_overt_message(embedded_posits_loaded, bit_size, nan_handling)
    write_reconstructed_overt_message(reconstructed_overt_message_decimal, bit_size, file_paths)
    reconstructed_overt_binary = readlines(file_paths["reconstructed_binary"])

    # Step 6: Analysis and Logging
    percentage_match_overt,
    percentage_match_covert,
    entropy_overt,
    entropy_modified,
    entropy_reconstructed,
    overt_byte_entropy,
    modified_byte_entropy,
    reconstructed_byte_entropy = perform_analysis(
        overt_message_binary,
        reconstructed_overt_binary,
        covert_message,
        extracted_covert_data,
        modified_message_binary,
        bit_size
    )

    write_log(
        bit_size,
        overt_message_length,
        actual_covert_length,
        config,
        percentage_match_overt,
        percentage_match_covert,
        entropy_overt,
        entropy_modified,
        entropy_reconstructed,
        overt_byte_entropy,
        modified_byte_entropy,
        reconstructed_byte_entropy,
        file_paths
    )

    println("Covert channel simulation completed successfully.")
end

# Run main function
main()
