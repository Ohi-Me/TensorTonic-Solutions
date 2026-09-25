def mean_rating_imputation(ratings_matrix: list, mode: str) -> list:

    # Make a copy of the original matrix
    result = []

    for row in ratings_matrix:
        result.append(row[:])

    # User mode
    if mode == "user":

        for i in range(len(ratings_matrix)):

            total = 0
            count = 0

            # Calculate mean of this user's ratings
            for j in range(len(ratings_matrix[i])):

                if ratings_matrix[i][j] != 0:
                    total += ratings_matrix[i][j]
                    count += 1

            if count > 0:
                mean = total / count
            else:
                mean = 0.0

            # Replace missing ratings
            for j in range(len(ratings_matrix[i])):

                if ratings_matrix[i][j] == 0:
                    result[i][j] = mean

    # Item mode
    else:

        for j in range(len(ratings_matrix[0])):

            total = 0
            count = 0

            # Calculate mean of this item's ratings
            for i in range(len(ratings_matrix)):

                if ratings_matrix[i][j] != 0:
                    total += ratings_matrix[i][j]
                    count += 1

            if count > 0:
                mean = total / count
            else:
                mean = 0.0

            # Replace missing ratings
            for i in range(len(ratings_matrix)):

                if ratings_matrix[i][j] == 0:
                    result[i][j] = mean

    return result