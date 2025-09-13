# utils/translations.py
import streamlit as st

# Translation dictionary
TRANSLATIONS = {
    'en': {
        # Navigation and titles
        'app_title': 'Data Analysis Tool',
        'import_clean': 'Import & Clean Data',
        'explore_correlate': 'Exploratory Data Analysis',
        'visualize_export': 'Visualization',
        'ks_test': 'Kolmogorov–Smirnov Test',
        
        # Import page
        'upload_dataset': 'Upload your dataset (.csv or .xlsx)',
        'data_preview': 'Data Preview',
        'data_validation': 'Data Validation',
        'data_cleaning': 'Data Cleaning Options',
        'cleaned_data': 'Cleaned Data Preview',
        'drop_duplicates': 'Drop duplicate rows',
        'handle_missing': 'Handle missing values',
        'do_nothing': 'Do nothing',
        'drop_rows_missing': 'Drop rows with missing values',
        'drop_cols_missing': 'Drop columns with missing values',
        'fill_mean': 'Fill with mean',
        'fill_median': 'Fill with median',
        'fill_mode': 'Fill with mode',
        'success_import': 'Data imported and cleaned successfully! You can now proceed to EDA.',
        'upload_file_prompt': 'Please upload a CSV or Excel file to get started.',
        
        # EDA page
        'descriptive_stats': 'Descriptive Statistics',
        'select_variables_desc': 'Select variables for descriptive stats',
        'download_desc_stats': 'Download descriptive stats',
        'normality_tests': 'Normality Tests',
        'select_numeric_normality': 'Select numeric columns for normality testing',
        'correlation_analysis': 'Correlation Analysis',
        'select_variables_corr': 'Select variables for correlation analysis',
        'correlation_method': 'Select correlation method',
        'compute_correlation': 'Compute correlation',
        'correlation_matrix': 'Correlation Matrix (rounded)',
        'pvalues_matrix': 'P-values Matrix (rounded)',
        'correlation_summary': 'Correlation Summary',
        'top_correlated_pairs': 'Top Correlated Pairs:',
        'total_pairs': 'Total Pairs',
        'significant_pairs': 'Significant Pairs',
        'mean_correlation': 'Mean Correlation',
        'max_correlation': 'Max Correlation',
        'normal_distributions': 'Normal Distributions',
        'non_normal_distributions': 'Non-Normal Distributions',
        'download_correlation': 'Download correlation matrix',
        'download_pvalues': 'Download p-values matrix',
        
        # Visualization page
        'visualization': 'Visualization',
        'correlation_heatmap': 'Correlation heatmap',
        'coefficient_plot': 'Coefficient plot',
        'custom_scatter': 'Custom scatter plot with regression line',
        'x_variable': 'X variable',
        'y_variable': 'Y variable',
        'run_correlation_first': 'Run correlation on the EDA page first to see a heatmap.',
        'fit_model_first': 'Fit a model in the Modeling page to generate coefficient plots.',
        'bundle_figures': 'Bundle figures and download (zip)',
        'download_figures': 'Download figures bundle',
        
        # KS Test page
        'ks_test_title': 'Kolmogorov–Smirnov Test',
        'test_configuration': 'Test Configuration',
        'select_column_ks': 'Select a column for KS Test:',
        'distribution_test': 'Distribution to test against:',
        'sample_size': 'Sample Size',
        'mean': 'Mean',
        'std_dev': 'Std Dev',
        'run_test': 'Run Test',
        'run_ks_test': 'Run KS Test',
        'ks_completed': 'KS Test completed',
        'test_results': 'Test Results',
        'ks_statistic': 'KS Statistic',
        'p_value': 'P-value',
        'conclusion': 'Conclusion',
        'interpretation': 'Interpretation',
        'null_hypothesis': 'H₀ (Null Hypothesis): The sample follows a {dist} distribution',
        'alt_hypothesis': 'H₁ (Alternative Hypothesis): The sample does not follow a {dist} distribution',
        'significance_level': 'Significance Level: α = 0.05',
        'export_results': 'Export Results',
        'download_pdf': 'Download Results as PDF',
        'batch_testing': 'Batch Testing',
        'enable_batch': 'Enable batch testing for multiple columns',
        'select_batch_cols': 'Select columns for batch KS testing:',
        'run_batch_tests': 'Run Batch KS Tests',
        'batch_completed': 'Batch testing completed for {count} columns',
        'batch_summary': 'Batch Results Summary',
        'follows_distribution': 'Follows Distribution',
        'download_batch_pdf': 'Download Batch Results as PDF',
        
        # Common messages
        'no_data': 'No dataset found. Please upload and clean a file on the Import page.',
        'dataset_info': 'Dataset Info: {rows} rows, {cols} columns',
        'rows': 'Rows',
        'columns': 'Columns',
        'error': 'Error',
        'warning': 'Warning',
        'success': 'Success',
        'info': 'Info',
        'yes': 'Yes',
        'no': 'No',
        'column': 'Column',
        'test': 'Test',
        'statistic': 'Statistic',
        'is_normal': 'Is Normal',
    },
    
    'ru': {
        # Navigation and titles
        'app_title': 'Инструмент анализа данных',
        'import_clean': 'Импорт и очистка данных',
        'explore_correlate': 'Исследовательский анализ данных',
        'visualize_export': 'Визуализация',
        'ks_test': 'Тест Колмогорова–Смирнова',
        
        # Import page
        'upload_dataset': 'Загрузите ваш набор данных (.csv или .xlsx)',
        'data_preview': 'Предварительный просмотр данных',
        'data_validation': 'Проверка данных',
        'data_cleaning': 'Опции очистки данных',
        'cleaned_data': 'Предварительный просмотр очищенных данных',
        'drop_duplicates': 'Удалить дублирующиеся строки',
        'handle_missing': 'Обработать пропущенные значения',
        'do_nothing': 'Ничего не делать',
        'drop_rows_missing': 'Удалить строки с пропущенными значениями',
        'drop_cols_missing': 'Удалить столбцы с пропущенными значениями',
        'fill_mean': 'Заполнить средним значением',
        'fill_median': 'Заполнить медианой',
        'fill_mode': 'Заполнить модой',
        'success_import': 'Данные успешно импортированы и очищены! Теперь вы можете перейти к EDA.',
        'upload_file_prompt': 'Пожалуйста, загрузите CSV или Excel файл для начала работы.',
        
        # EDA page
        'descriptive_stats': 'Описательная статистика',
        'select_variables_desc': 'Выберите переменные для описательной статистики',
        'download_desc_stats': 'Скачать описательную статистику',
        'normality_tests': 'Тесты нормальности',
        'select_numeric_normality': 'Выберите числовые столбцы для тестирования нормальности',
        'correlation_analysis': 'Корреляционный анализ',
        'select_variables_corr': 'Выберите переменные для корреляционного анализа',
        'correlation_method': 'Выберите метод корреляции',
        'compute_correlation': 'Вычислить корреляцию',
        'correlation_matrix': 'Матрица корреляции (округленная)',
        'pvalues_matrix': 'Матрица p-значений (округленная)',
        'correlation_summary': 'Сводка корреляции',
        'top_correlated_pairs': 'Топ коррелированных пар:',
        'total_pairs': 'Всего пар',
        'significant_pairs': 'Значимых пар',
        'mean_correlation': 'Средняя корреляция',
        'max_correlation': 'Максимальная корреляция',
        'normal_distributions': 'Нормальные распределения',
        'non_normal_distributions': 'Ненормальные распределения',
        'download_correlation': 'Скачать матрицу корреляции',
        'download_pvalues': 'Скачать матрицу p-значений',
        
        # Visualization page
        'visualization': 'Визуализация',
        'correlation_heatmap': 'Тепловая карта корреляции',
        'coefficient_plot': 'График коэффициентов',
        'custom_scatter': 'Пользовательский точечный график с линией регрессии',
        'x_variable': 'Переменная X',
        'y_variable': 'Переменная Y',
        'run_correlation_first': 'Сначала запустите корреляцию на странице EDA, чтобы увидеть тепловую карту.',
        'fit_model_first': 'Подгоните модель на странице моделирования для создания графиков коэффициентов.',
        'bundle_figures': 'Собрать графики и скачать (zip)',
        'download_figures': 'Скачать пакет графиков',
        
        # KS Test page
        'ks_test_title': 'Тест Колмогорова–Смирнова',
        'test_configuration': 'Настройка теста',
        'select_column_ks': 'Выберите столбец для KS-теста:',
        'distribution_test': 'Распределение для тестирования:',
        'sample_size': 'Размер выборки',
        'mean': 'Среднее',
        'std_dev': 'Стд. откл.',
        'run_test': 'Запустить тест',
        'run_ks_test': 'Запустить KS-тест',
        'ks_completed': 'KS-тест завершен',
        'test_results': 'Результаты теста',
        'ks_statistic': 'KS-статистика',
        'p_value': 'P-значение',
        'conclusion': 'Заключение',
        'interpretation': 'Интерпретация',
        'null_hypothesis': 'H₀ (Нулевая гипотеза): Выборка следует {dist} распределению',
        'alt_hypothesis': 'H₁ (Альтернативная гипотеза): Выборка не следует {dist} распределению',
        'significance_level': 'Уровень значимости: α = 0.05',
        'export_results': 'Экспорт результатов',
        'download_pdf': 'Скачать результаты в PDF',
        'batch_testing': 'Пакетное тестирование',
        'enable_batch': 'Включить пакетное тестирование для нескольких столбцов',
        'select_batch_cols': 'Выберите столбцы для пакетного KS-тестирования:',
        'run_batch_tests': 'Запустить пакетные KS-тесты',
        'batch_completed': 'Пакетное тестирование завершено для {count} столбцов',
        'batch_summary': 'Сводка пакетных результатов',
        'follows_distribution': 'Следует распределению',
        'download_batch_pdf': 'Скачать пакетные результаты в PDF',
        
        # Common messages
        'no_data': 'Набор данных не найден. Пожалуйста, загрузите и очистите файл на странице импорта.',
        'dataset_info': 'Информация о наборе данных: {rows} строк, {cols} столбцов',
        'rows': 'Строки',
        'columns': 'Столбцы',
        'error': 'Ошибка',
        'warning': 'Предупреждение',
        'success': 'Успех',
        'info': 'Информация',
        'yes': 'Да',
        'no': 'Нет',
        'column': 'Столбец',
        'test': 'Тест',
        'statistic': 'Статистика',
        'is_normal': 'Нормальное',
    }
}

def get_text(key: str, **kwargs) -> str:
    """
    Get translated text based on current language setting
    
    Args:
        key: Translation key
        **kwargs: Format arguments for string formatting
        
    Returns:
        Translated text
    """
    # Get current language from session state
    lang = st.session_state.get('language', 'en')
    
    # Get translation
    text = TRANSLATIONS.get(lang, TRANSLATIONS['en']).get(key, key)
    
    # Format with kwargs if provided
    if kwargs:
        try:
            text = text.format(**kwargs)
        except:
            pass  # If formatting fails, return original text
    
    return text

def language_selector():
    """
    Create language selector widget in sidebar
    """
    st.sidebar.markdown("---")
    st.sidebar.subheader("🌐 Language / Язык")
    
    languages = {
        'en': '🇺🇸 English',
        'ru': '🇷🇺 Русский'
    }
    
    current_lang = st.session_state.get('language', 'en')
    
    selected_lang = st.sidebar.selectbox(
        "Select Language:",
        options=list(languages.keys()),
        format_func=lambda x: languages[x],
        index=list(languages.keys()).index(current_lang),
        key="language_selector"
    )
    
    if selected_lang != current_lang:
        st.session_state['language'] = selected_lang
        st.rerun()

def init_language():
    """
    Initialize language setting
    """
    if 'language' not in st.session_state:
        st.session_state['language'] = 'en'