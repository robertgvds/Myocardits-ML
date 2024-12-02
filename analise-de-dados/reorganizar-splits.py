def divide_into_groups(individuals, num_groups, target_sum):
    # Cria grupos vazios e suas somas
    groups = [[] for _ in range(num_groups)]
    group_sums = [0] * num_groups

    # Ordena os indivíduos pela soma em ordem decrescente
    sorted_individuals = sorted(individuals, key=lambda x: x[1], reverse=True)

    # Atribui cada indivíduo ao grupo com a menor soma atual
    for individual in sorted_individuals:
        # Encontra o grupo com a menor soma atual
        min_group_index = group_sums.index(min(group_sums))
        groups[min_group_index].append(individual)
        group_sums[min_group_index] += individual[1]

    return groups, group_sums

# Lista de indivíduos com novas quantidades
individuals = [

("Individuo_17", 88),
("Individuo_18", 130),
("Individuo_19", 30),
("Individuo_20", 56),
("Individuo_21", 297),
("Individuo_22", 115),
("Individuo_23", 15),
("Individuo_24", 79),
("Individuo_25", 111),
("Individuo_26", 53),
("Individuo_27", 22),
("Individuo_28", 108),
("Individuo_29", 31),
("Individuo_30", 40),
("Individuo_31", 13),
("Individuo_32", 20),
("Individuo_33", 143),
("Individuo_34", 101),
("Individuo_35", 41),
("Individuo_36", 29),
("Individuo_37", 23),
("Individuo_38", 19),
("Individuo_39", 160),
("Individuo_40", 105),
("Individuo_41", 127),
("Individuo_42", 49),
("Individuo_43", 21),
("Individuo_44", 23),
("Individuo_45", 14),
("Individuo_46", 32),
("Individuo_47", 17),
]



# Média desejada das somas
target_sum = 1335
num_groups = 5

# Dividindo os indivíduos em grupos
groups, group_sums = divide_into_groups(individuals, num_groups, target_sum)

# Exibindo os resultados
for i, group in enumerate(groups):
    group_names = [ind[0] for ind in group]
    print(f"Grupo {i + 1}: {group_names}, Soma: {group_sums[i]}")

# Exibindo a média das somas dos grupos
print(f"Média das somas dos grupos: {sum(group_sums) / num_groups:.2f}")
