import csv

data = """## 0.2 Validation split
ThreeClassSub1_delta, 0.4060150384902954
ThreeClassSub1_alpha, 0.3834586441516876
ThreeClassSub1_beta, 0.4285714328289032
ThreeClassSub1_deltaAlpha, 0.31578946113586426
ThreeClassSub1_deltaBeta, 0.3834586441516876
ThreeClassSub1_alphaBeta, 0.4135338366031647
ThreeClassSub1_deltaalphaBeta, 0.4060150384902954
ThreeClassSub2_delta, 0.5680000185966492
ThreeClassSub2_alpha, 0.6399999856948853
ThreeClassSub2_beta, 0.5839999914169312
ThreeClassSub2_deltaAlpha, 0.6079999804496765
ThreeClassSub2_deltaBeta, 0.7200000286102295
ThreeClassSub2_alphaBeta, 0.6399999856948853
ThreeClassSub2_deltaalphaBeta, 0.6639999747276306
TwoClassSub1_delta_onebacktwoback, 0.6404494643211365
TwoClassSub1_delta_onebacktwoback, 0.6067415475845337
TwoClassSub1_delta_onebacktwoback, 0.5340909361839294
TwoClassSub1_alpha_onebacktwoback, 0.617977499961853
TwoClassSub1_alpha_onebacktwoback, 0.5617977380752563
TwoClassSub1_alpha_onebacktwoback, 0.5454545617103577
TwoClassSub1_beta_onebacktwoback, 0.5730336904525757
TwoClassSub1_beta_onebacktwoback, 0.617977499961853
TwoClassSub1_beta_onebacktwoback, 0.5909090638160706
TwoClassSub1_delta_alpha_onebacktwoback, 0.5955055952072144
TwoClassSub1_delta_alpha_onebacktwoback, 0.617977499961853
TwoClassSub1_delta_alpha_onebacktwoback, 0.4886363744735718
TwoClassSub1_delta_beta_onebacktwoback, 0.584269642829895
TwoClassSub1_delta_beta_onebacktwoback, 0.617977499961853
TwoClassSub1_delta_beta_onebacktwoback, 0.5
TwoClassSub1_alpha_beta_onebacktwoback, 0.6404494643211365
TwoClassSub1_alpha_beta_onebacktwoback, 0.6516854166984558
TwoClassSub1_alpha_beta_onebacktwoback, 0.5340909361839294
TwoClassSub1_all_onebacktwoback, 0.6516854166984558
TwoClassSub1_all_onebacktwoback, 0.6292135119438171
TwoClassSub1_all_onebacktwoback, 0.5568181872367859
TwoClassSub2_delta_onebacktwoback, 0.7710843086242676
TwoClassSub2_delta_onebacktwoback, 0.6024096608161926
TwoClassSub2_delta_onebacktwoback, 0.773809552192688
TwoClassSub2_alpha_onebacktwoback, 0.6867470145225525
TwoClassSub2_alpha_onebacktwoback, 0.759036123752594
TwoClassSub2_alpha_onebacktwoback, 0.8452380895614624
TwoClassSub2_beta_onebacktwoback, 0.759036123752594
TwoClassSub2_beta_onebacktwoback, 0.6987951993942261
TwoClassSub2_beta_onebacktwoback, 0.8214285969734192
TwoClassSub2_delta_alpha_onebacktwoback, 0.7831325531005859
TwoClassSub2_delta_alpha_onebacktwoback, 0.6626505851745605
TwoClassSub2_delta_alpha_onebacktwoback, 0.8095238208770752
TwoClassSub2_delta_beta_onebacktwoback, 0.759036123752594
TwoClassSub2_delta_beta_onebacktwoback, 0.7108433842658997
TwoClassSub2_delta_beta_onebacktwoback, 0.8333333134651184
TwoClassSub2_alpha_beta_onebacktwoback, 0.7349397540092468
TwoClassSub2_alpha_beta_onebacktwoback, 0.7349397540092468
TwoClassSub2_alpha_beta_onebacktwoback, 0.8333333134651184
TwoClassSub2_all_onebacktwoback, 0.6867470145225525
TwoClassSub2_all_onebacktwoback, 0.7108433842658997
TwoClassSub2_all_onebacktwoback, 0.8928571343421936
"""
lines = data.split("\n")[1:]  # Exclude the first line (comment line)

# Parsing the data
lines = data.split("\n")[1:]  # Exclude the first line (comment line)

# Create CSV
with open('data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Description", "Subject", "Band Combination", "Task Combination", "Accuracy Score"])  # Writing header

    for line_number, line in enumerate(lines, start=1):
        try:
            # Split data and task_combination
            desc_score = line.split(", ")
            desc = desc_score[0]
            score = round(float(desc_score[1]), 3)

            # Split description into parts
            task_comb = "-"
            if "_" in desc:
                desc_parts = desc.split("_")
                task_comb = desc_parts[-1] if len(desc_parts) == 3 else "-"
                desc = "_".join(desc_parts[:-1]) if len(desc_parts) == 3 else desc

            # Extract subject
            subject = desc[-1]
            desc = desc[:-1]  # Remove subject number from description

            # Extract bands
            bands = desc.split("_")[-1]
            desc = "_".join(desc.split("_")[:-1])  # Remove bands from description

            # Handle concatenated bands
            band_comb = []
            for band in ["delta", "alpha", "beta"]:
                if band in bands:
                    band_comb.append(band)
                    bands = bands.replace(band, "")
            band_comb = ", ".join(band_comb)

            # Write row to CSV
            writer.writerow([desc, subject, band_comb, task_comb, score])
        except IndexError:
            print(f"Error: Unable to parse line {line_number}: {line}. Skipping this line.")
            continue
        except ValueError:
            print(f"Error: Unable to convert score to float on line {line_number}: {line}. Skipping this line.")
            continue
        except Exception as e:
            print(f"Unexpected error at line {line_number}: {str(e)}. Skipping this line.")
            continue
