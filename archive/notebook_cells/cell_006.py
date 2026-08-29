# ============================================
# Cell 7. Define conditions and perturbation operators M_q
# ============================================

# Current perturbation set after the first two QC steps
final_perturbations = filtered_perturbations.copy()

# q = 0 is the non-targeting control
final_conditions = ["non-targeting"] + final_perturbations

# For now, use the perturbation targets themselves
# as the candidate regulator set R
regulator_genes = final_perturbations.copy()

K = len(regulator_genes)
Q = len(final_conditions)

regulator_to_index = {
    gene: j for j, gene in enumerate(regulator_genes)
}

condition_to_index = {
    condition: q for q, condition in enumerate(final_conditions)
}


def make_Mq(condition):
    """
    Perturbation operator M_q for the manuscript model.

    Control:
        M_0 = I

    Target perturbation:
        zero the corresponding regulator coordinate.
    """
    M = np.eye(K)

    if condition != "non-targeting":
        j = regulator_to_index[condition]
        M[j, j] = 0.0

    return M


print("Candidate regulators K:", K)
print("Conditions including control:", Q)

# Check control
M0 = make_Mq("non-targeting")
print("\nControl M0 is identity:", np.allclose(M0, np.eye(K)))

# Check one perturbation
test_gene = "TACC3"
Mq_test = make_Mq(test_gene)
j = regulator_to_index[test_gene]

print("\nTest perturbation:", test_gene)
print("Target diagonal entry:", Mq_test[j, j])
print("Number of zero diagonal entries:", np.sum(np.diag(Mq_test) == 0))
print("All other diagonal entries = 1:",
      np.all(np.delete(np.diag(Mq_test), j) == 1))