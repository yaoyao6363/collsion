import os

import xlsxwriter


def custom_collate(batch: list) -> list:
    return [item for item in batch]


def save_metrics(dir_: str, artifacts: dict):
    """
    Save the artifacts into a xlsx file
    :param dir_: Directory path
    :param artifacts: Dictionary containing all artifacts
    :return: None
    """
    workbook = xlsxwriter.Workbook(os.path.join(dir_, 'artifacts.xlsx'))
    worksheet = workbook.add_worksheet()
    row, col = 0, 0
    for key in artifacts.keys():
        row = 1
        col += 1
        worksheet.write(row, col, key)
        for item in artifacts[key]:
            row += 1
            worksheet.write(row, col, item)
    workbook.close()
