import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors3D
import tensorflow as tf
import joblib
import os

# Define paths
model_path = 'D:\cof_model\model_ANN-best_7.keras'
scaler_X_path = 'D:\cof_model\scaler_x.pkl'
scaler_y_path = 'D:\cof_model\scaler_y.pkl'


# Define paths


def load_model_and_scalers():
    try:
        model = tf.keras.models.load_model(model_path)
        scaler_X = joblib.load(scaler_X_path)
        scaler_y = joblib.load(scaler_y_path)
        return model, scaler_X, scaler_y
    except Exception as e:
        st.error(f"Error loading model or scalers: {e}")
        return None, None, None

def check_file_exists(file_path):
    return os.path.isfile(file_path)

# Check if model file exists
if not check_file_exists(model_path):
    st.error(f"File not found: {model_path}")

# Define the experimental parameters list
Exp_list = ['Cat_mg', 'CoCat_wt%', 'CoCat_\nRu(bpy)3Cl2', 'CoCat_Co(NO3)2', 'CoCat_Cu3(HHTP)2', 
            'CoCat_H2PtCl6', 'CoCat_HAuCl4', 'CoCat_Ni(OAc)', 'CoCat_Ni(OH)2', 'CoCat_Non', 
            'CoCat_PVP-Pt', 'CoCat_Pt', 'CoCat_none', 'SED_AA', 'SED_L-Ascorbic', 'SED_L-Cystein', 
            'SED_MeOH', 'SED_Na2S-Na2SO3', 'SED_SA', 'SED_TEA', 'SED_TEOA', 'SED_VC', 'SED_none']

# Initialize DataFrame with zeroes
df_exp = pd.DataFrame(0, index=range(1), columns=Exp_list)

# Define a function to calculate descriptors
def calculate_descriptors(smiles):
    fingerprint_descriptors_list = []
    descriptors_3d_list = []

    mol = Chem.MolFromSmiles(smiles)
    if mol:
        # Calculate Morgan fingerprint descriptors
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2)  # Morgan fingerprint with radius 2
        fingerprint_descriptors_list.append(list(fingerprint))

        # Generate 3D conformers
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, useRandomCoords=True)
        AllChem.UFFOptimizeMolecule(mol)

        # Calculate 3D descriptors
        descriptors_3d = {
            "Asphericity": Descriptors3D.Asphericity(mol),
            "Eccentricity": Descriptors3D.Eccentricity(mol),
            "InertialShapeFactor": Descriptors3D.InertialShapeFactor(mol),
            "PMI1": Descriptors3D.PMI1(mol),
            "PMI2": Descriptors3D.PMI2(mol),
            "PMI3": Descriptors3D.PMI3(mol),
            "RadiusOfGyration": Descriptors3D.RadiusOfGyration(mol),
            "SpherocityIndex": Descriptors3D.SpherocityIndex(mol),
            "NPR1": Descriptors3D.NPR1(mol),
            "NPR2": Descriptors3D.NPR2(mol),
            "PBF": Descriptors3D.PBF(mol)
        }
        descriptors_3d_list.append(descriptors_3d)

    # Convert the lists of descriptors into DataFrames
    fingerprint_descriptors_df = pd.DataFrame(fingerprint_descriptors_list)
    descriptors_3d_df = pd.DataFrame(descriptors_3d_list)

    # Concatenate the descriptors DataFrames (since there's no original dataset)
    result_combined = pd.concat([fingerprint_descriptors_df, descriptors_3d_df], axis=1)

    # Define your selected list of columns
    #Selected_list = ['6', '13', '18', '20', '21', '24', '25', '33', '34', '36', '45', '47', '57', '58', '62', '66', '67', '71', '74', '80', '82', '84', '86', '91', '92', '102', '108', '119', '122', '126', '128', '131', '137', '140', '146', '149', '157', '161', '162', '164', '168', '170', '174', '176', '180', '183', '185', '186', '187', '189', '191', '193', '196', '200', '202', '207', '208', '212', '213', '222', '224', '233', '235', '237', '239', '243', '247', '249', '252', '261', '266', '270', '271', '281', '284', '287', '290', '294', '296', '303', '305', '309', '310', '312', '314', '318', '319', '320', '322', '325', '335', '338', '343', '344', '345', '350', '352', '354', '358', '359', '361', '372', '373', '376', '378', '382', '383', '389', '400', '407', '411', '421', '423', '425', '426', '430', '441', '443', '447', '453', '456', '463', '465', '469', '472', '475', '478', '479', '480', '486', '492', '494', '503', '504', '510', '511', '512', '539', '541', '542', '544', '547', '550', '552', '558', '561', '568', '570', '584', '585', '587', '588', '590', '591', '593', '597', '606', '612', '621', '623', '624', '626', '636', '638', '640', '647', '650', '656', '658', '660', '665', '668', '670', '674', '675', '680', '681', '683', '686', '687', '690', '694', '695', '699', '702', '711', '713', '714', '715', '716', '718', '723', '724', '725', '727', '728', '734', '736', '740', '745', '747', '749', '753', '764', '765', '767', '769', '773', '781', '785', '786', '789', '790', '794', '799', '803', '804', '806', '807', '816', '820', '824', '828', '834', '835', '838', '840', '841', '842', '843', '845', '848', '853', '855', '856', '864', '868', '870', '871', '875', '878', '879', '880', '881', '885', '888', '890', '893', '896', '897', '902', '903', '906', '909', '910', '913', '914', '915', '916', '922', '924', '926', '930', '931', '932', '935', '944', '945', '948', '956', '957', '960', '961', '963', '964', '973', '974', '979', '980', '982', '984', '988', '1003', '1004', '1011', '1012', '1015', '1017', '1019', '1021', '1025', '1039', '1047', '1048', '1050', '1056', '1057', '1059', '1060', '1063', '1065', '1073', '1074', '1079', '1081', '1084', '1088', '1089', '1096', '1097', '1104', '1106', '1110', '1113', '1114', '1118', '1119', '1120', '1123', '1126', '1128', '1132', '1134', '1136', '1140', '1143', '1145', '1147', '1152', '1154', '1155', '1156', '1157', '1160', '1167', '1168', '1169', '1171', '1173', '1174', '1179', '1190', '1192', '1193', '1195', '1199', '1200', '1203', '1205', '1208', '1213', '1221', '1229', '1234', '1237', '1238', '1240', '1246', '1247', '1250', '1252', '1254', '1255', '1256', '1257', '1259', '1265', '1268', '1274', '1279', '1280', '1282', '1295', '1302', '1309', '1313', '1315', '1323', '1326', '1328', '1330', '1333', '1344', '1347', '1350', '1352', '1355', '1357', '1359', '1360', '1363', '1366', '1368', '1371', '1373', '1374', '1380', '1381', '1382', '1384', '1385', '1390', '1392', '1396', '1398', '1399', '1401', '1404', '1411', '1414', '1420', '1421', '1423', '1432', '1433', '1436', '1440', '1445', '1452', '1455', '1461', '1462', '1467', '1472', '1474', '1476', '1477', '1481', '1485', '1487', '1489', '1491', '1494', '1497', '1502', '1504', '1505', '1508', '1515', '1517', '1524', '1530', '1534', '1537', '1539', '1541', '1543', '1544', '1553', '1555', '1556', '1560', '1566', '1570', '1571', '1572', '1575', '1578', '1581', '1582', '1583', '1585', '1588', '1590', '1592', '1594', '1596', '1597', '1599', '1602', '1603', '1604', '1607', '1609', '1613', '1617', '1618', '1619', '1620', '1623', '1625', '1626', '1627', '1629', '1631', '1635', '1636', '1640', '1642', '1644', '1645', '1648', '1649', '1653', '1654', '1666', '1669', '1673', '1679', '1681', '1683', '1684', '1686', '1697', '1698', '1701', '1708', '1722', '1724', '1727', '1728', '1733', '1737', '1738', '1739', '1743', '1745', '1746', '1747', '1750', '1754', '1762', '1768', '1773', '1776', '1781', '1784', '1793', '1794', '1799', '1800', '1801', '1804', '1807', '1808', '1812', '1814', '1818', '1819', '1823', '1825', '1826', '1827', '1840', '1842', '1846', '1848', '1849', '1854', '1855', '1858', '1862', '1866', '1867', '1871', '1876', '1879', '1885', '1889', '1893', '1896', '1897', '1911', '1912', '1915', '1916', '1917', '1918', '1920', '1921', '1924', '1925', '1928', '1930', '1934', '1939', '1946', '1947', '1948', '1949', '1951', '1953', '1954', '1963', '1964', '1969', '1970', '1971', '1980', '1983', '1984', '1985', '1987', '1991', '1992', '1993', '1994', '1998', '2001', '2003', '2004', '2007', '2013', '2021', '2024', '2032', '2033']
    Selected_list = ['6', '13', '18', '20', '21', '24', '25', '33', '34', '36', '45', '47', '57', '58', '62', '66', '67', '71', '74', '80', '82', '84', '86', '91', '92', '102', '108', '119', '122', '126', '128', '131', '137', '140', '146', '149', '157', '161', '162', '164', '168', '170', '174', '176', '180', '183', '185', '186', '187', '189', '191', '193', '196', '200', '202', '207', '208', '212', '213', '222', '224', '233', '235', '237', '239', '243', '247', '249', '252', '261', '266', '270', '271', '281', '284', '287', '290', '294', '296', '303', '305', '309', '310', '312', '314', '318', '319', '320', '322', '325', '335', '338', '343', '344', '345', '350', '352', '354', '358', '359', '361', '372', '373', '376', '378', '382', '383', '389', '400', '407', '411', '421', '423', '425', '426', '430', '441', '443', '447', '453', '456', '463', '465', '469', '472', '475', '478', '479', '480', '486', '492', '494', '503', '504', '510', '511', '512', '539', '541', '542', '544', '547', '550', '552', '558', '561', '568', '570', '584', '585', '587', '588', '590', '591', '593', '597', '606', '612', '621', '623', '624', '626', '636', '638', '640', '647', '650', '656', '658', '660', '665', '668', '670', '674', '675', '680', '681', '683', '686', '687', '690', '694', '695', '699', '702', '711', '713', '714', '715', '716', '718', '723', '724', '725', '727', '728', '734', '736', '740', '745', '747', '749', '753', '764', '765', '767', '769', '773', '781', '785', '786', '789', '790', '794', '799', '803', '804', '806', '807', '816', '820', '824', '828', '834', '835', '838', '840', '841', '842', '843', '845', '848', '853', '855', '856', '864', '868', '870', '871', '875', '878', '879', '880', '881', '885', '888', '890', '893', '896', '897', '902', '903', '906', '909', '910', '913', '914', '915', '916', '922', '924', '926', '930', '931', '932', '935', '944', '945', '948', '956', '957', '960', '961', '963', '964', '973', '974', '979', '980', '982', '984', '988', '1003', '1004', '1011', '1012', '1015', '1017', '1019', '1021', '1025', '1039', '1047', '1048', '1050', '1056', '1057', '1059', '1060', '1063', '1065', '1073', '1074', '1079', '1081', '1084', '1088', '1089', '1096', '1097', '1104', '1106', '1110', '1113', '1114', '1118', '1119', '1120', '1123', '1126', '1128', '1132', '1134', '1136', '1140', '1143', '1145', '1147', '1152', '1154', '1155', '1156', '1157', '1160', '1167', '1168', '1169', '1171', '1173', '1174', '1179', '1190', '1192', '1193', '1195', '1199', '1200', '1203', '1205', '1208', '1213', '1221', '1229', '1234', '1237', '1238', '1240', '1246', '1247', '1250', '1252', '1254', '1255', '1256', '1257', '1259', '1265', '1268', '1274', '1279', '1280', '1282', '1295', '1302', '1309', '1313', '1315', '1323', '1326', '1328', '1330', '1333', '1344', '1347', '1350', '1352', '1355', '1357', '1361', '1365', '1369', '1370', '1372', '1379', '1384', '1385', '1391', '1395', '1400', '1405', '1407', '1410', '1415', '1418', '1425', '1432', '1435', '1438', '1440', '1441', '1444', '1445', '1452', '1454', '1457', '1460', '1462', '1468', '1476', '1477', '1479', '1485', '1487', '1491', '1516', '1530', '1533', '1535', '1536', '1541', '1543', '1548', '1551', '1557', '1563', '1565', '1572', '1574', '1582', '1602', '1603', '1604', '1607', '1609', '1613', '1617', '1618', '1619', '1620', '1623', '1625', '1626', '1627', '1629', '1631', '1635', '1636', '1640', '1642', '1644', '1645', '1648', '1649', '1653', '1654', '1666', '1669', '1673', '1679', '1681', '1683', '1684', '1686', '1697', '1698', '1701', '1708', '1722', '1724', '1727', '1728', '1733', '1737', '1738', '1739', '1743', '1745', '1746', '1747', '1750', '1754', '1762', '1768', '1773', '1776', '1781', '1784', '1793', '1794', '1799', '1800', '1801', '1804', '1807', '1808', '1812', '1814', '1818', '1819', '1823', '1825', '1826', '1827', '1840', '1842', '1846', '1848', '1849', '1854', '1855', '1858', '1862', '1866', '1867', '1871', '1876', '1879', '1885', '1889', '1893', '1896', '1897', '1911', '1912', '1915', '1916', '1917', '1918', '1920', '1921', '1924', '1925', '1928', '1930', '1934', '1939', '1946', '1947', '1948', '1949', '1951', '1953', '1954', '1963', '1964', '1969', '1970', '1971', '1980', '1983', '1984', '1985', '1987', '1991', '1992', '1993', '1994', '1998', '2001', '2003', '2004', '2007', '2013', '2021', '2024', '2032', '2033']

    # Ensure column names are strings
    result_combined.columns = result_combined.columns.astype(str)

    # Filter columns in result_combined based on Selected_list
    numeric_columns_to_keep = [col for col in Selected_list if col.isdigit()]

    # Filter result_combined to include only columns in numeric_columns_to_keep
    result_combined_filtered = result_combined[numeric_columns_to_keep]
    result_combined2 = pd.concat([result_combined_filtered, descriptors_3d_df], axis=1)

    return result_combined2

# Define the pages
def page_one():
    st.title("COF-H2 Predictor Platform")
    st.write("This page contains various input fields.")

    # Text input box
    smile_input = st.text_input("Enter SMILE Structure:")

    # Two select boxes
    option1 = st.selectbox("Choose CoCat:", ["CoCat_\nRu(bpy)3Cl2", "CoCat_Co(NO3)2", "CoCat_Cu3(HHTP)2", "CoCat_H2PtCl6", "CoCat_HAuCl4", "CoCat_Ni(OAc)", "CoCat_Ni(OH)2", "CoCat_Non", "CoCat_PVP-Pt", "CoCat_Pt", "CoCat_none"])
    option2 = st.selectbox("Choose SED:", ["SED_AA", "SED_L-Ascorbic", "SED_L-Cystein", "SED_MeOH", "SED_Na2S-Na2SO3", "SED_SA", "SED_TEA", "SED_TEOA", "SED_VC", "SED_none"])

    # Slider inputs
    Cat_mg = st.slider("Cat_ weight (mg)?", 0, 50, 3)
    CoCat_wt = st.slider("CoCat_wt%?", 0, 50, 3)

    st.write("---------------------------------------------------------")
    st.markdown("<h2 style='color:green;'>Please be sure about the following inputs before Prediction</h2>", unsafe_allow_html=True)
    st.write(f"SMILE Structure: {smile_input}")
    st.write(f"Selected CoCat: {option1}")
    st.write(f"Selected SED: {option2}")
    st.write("Cat_ weight", Cat_mg, "mg")
    st.write("CoCat_wt%", CoCat_wt, "%")

    if st.button("Submit"):
        if smile_input:
            result_combined2 = calculate_descriptors(smile_input)

            # Update the experimental DataFrame
            df_exp.loc[0, 'Cat_mg'] = Cat_mg
            df_exp.loc[0, 'CoCat_wt%'] = CoCat_wt
            df_exp.loc[0, option1] = 1
            df_exp.loc[0, option2] = 1

            st.write("Descriptors Calculated:")
            st.dataframe(result_combined2)

            st.write("Experimental Parameters DataFrame:")
            st.dataframe(df_exp)
            
            input_df = pd.concat([df_exp, result_combined2], axis=1)
            st.write("All Parameters DataFrame:")
            st.dataframe(input_df)
        else:
            st.write("Please enter a SMILE Structure!")


    
        if 'result_combined2' in locals() and not result_combined2.empty:
            # Combine DataFrames
           # input_df = pd.concat([df_exp, result_combined2], axis=1)

            # Load the model and scalers
            try:
                model = tf.keras.models.load_model('D:\cof_model\model_ANN-best_7.keras')
                scaler_X = joblib.load('D:\cof_model\scaler_x.pkl')
                scaler_y = joblib.load('D:\cof_model\scaler_y.pkl')
            except Exception as e:
                st.write(f"Error loading model or scalers: {e}")
                return

            # Scale the input data
            scaled_input = scaler_X.transform(input_df)

            # Make predictions on the scaled input data
            scaled_predictions = model.predict(scaled_input)

            # Inverse transform the predictions to get them back to the original scale
            predictions = scaler_y.inverse_transform(scaled_predictions)

            st.write("Predictions:")
            st.write(predictions)
            st.write("μmol*h-1")
            #st.write("Error")
        else:
            st.write("Please complete the descriptors calculation before predicting.")


def page_two():
    st.title("Page 2: Optimization")
    st.write("This page can contain any additional content or functionality.")

def page_three():
    st.title("Page 3: Placeholder")
    st.write("This page can contain any additional content or functionality.")

# Create a sidebar with navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select a page:", ["Page 1", "Page 2", "Page 3"])

# Render the selected page
if page == "Page 1":
    page_one()
elif page == "Page 2":
    page_two()
elif page == "Page 3":
    page_three()
