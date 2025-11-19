import pandas as pd


path = "/Users/babajaguska/Documents/Buck_aging_papers_and_docs/FibroblastsDNA_dmg_Rapamycin/GSE248823_series_matrix.txt"
full_table_path = "/Users/babajaguska/Documents/Buck_aging_papers_and_docs/FibroblastsDNA_dmg_Rapamycin/GPL17586-45144.txt"


def test_read_microarray_file():
    """Test reading a microarray file into a DataFrame."""
    # Read the microarray data file
    df = pd.read_csv(path, sep="\t", comment="!")

    # Basic assertions to verify the DataFrame structure
    assert not df.empty, "DataFrame should not be empty"
    assert "ID_REF" in df.columns, "DataFrame should contain 'ID_REF' column"
    assert df.shape[1] > 1, "DataFrame should have more than one column"
    assert df.shape[0] > 1, "DataFrame should have more than one row"
    print(df.head())
    print("Looks ok.")
    return df


def inspect_full_table():
    """Inspect the full table file for additional information."""
    full_df = pd.read_csv(full_table_path, sep="\t", comment="#")

    full_df = full_df.rename(columns={"ID": "ID_REF"})

    print(full_df.head())

    print(f"Columns: {full_df.columns.tolist()}")
    print(f"Gene assignment preview: {full_df['gene_assignment'].head()}")

    full_df["Gene_Symbol"] = full_df["gene_assignment"].apply(
        extract_gene_symbol
    )

    # how many nan gene symbols?
    nan_count = full_df["Gene_Symbol"].isna().sum()
    print(f"Number of NaN Gene Symbols: {nan_count}")
    # remove
    full_df = full_df.dropna(subset=["Gene_Symbol"])
    print(full_df[["ID_REF", "Gene_Symbol"]].head())

    return full_df


def extract_gene_symbol(gene_assignment):
    """Extract gene symbol from gene assignment string."""
    if pd.isna(gene_assignment):
        return None
    parts = gene_assignment.split(" /// ")
    if parts:
        first_part = parts[0]
        symbols = first_part.split(" // ")
        if len(symbols) < 2:
            return None
        symbol = symbols[1].strip()
        if symbol == "":
            return None
        if symbol == "---":
            return None
        return symbol


def merge_dataframes(df_microarray, df_full):
    """Merge microarray DataFrame with full table DataFrame on 'ID_REF'."""
    print(f"DataFrame shapes: df={df.shape}, df_ref={df_ref.shape}")

    merged_df = pd.merge(
        df_microarray,
        df_full[["ID_REF", "Gene_Symbol"]],
        on="ID_REF",
        how="inner",
    )

    merged_df = merged_df.set_index("Gene_Symbol")
    merged_df = merged_df.drop(columns=["ID_REF"])
    print(merged_df.head())
    print(f"Merged DataFrame shape: {merged_df.shape}")
    return merged_df


df = test_read_microarray_file()
df_ref = inspect_full_table()
df_merged = merge_dataframes(df, df_ref)
