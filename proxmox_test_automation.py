#!/usr/bin/env python3
"""
Proxmox クラスターテスト自動化フレームワーク

このスクリプトは外部テストケースファイルを読み込み、Proxmox VE クラスターの
テストを自動化するためのフレームワークを提供します。
"""

import argparse
import csv
import json
import logging
import os
import paramiko
import re
import requests
import subprocess
import sys
import time
import uuid
import socket
import psutil
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from dotenv import load_dotenv

# SSLの警告を無視（必要に応じて）
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# .envファイルをロード
load_dotenv()

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("proxmox_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ProxmoxTester")

class TestFileParser:
    """テストファイル解析クラス"""
    
    @staticmethod
    def parse_file(file_path: str) -> List[Dict]:
        """
        テストケースファイルを解析
        
        Args:
            file_path: ファイルパス（CSVまたはTSV）
            
        Returns:
            テストケースのリスト
        """
        # ファイル拡張子からフォーマットを推定
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.csv':
            return TestFileParser._parse_csv(file_path)
        elif file_ext == '.tsv':
            return TestFileParser._parse_tsv(file_path)
        else:
            # 拡張子でわからない場合はファイル内容から判断
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                if '\t' in first_line:
                    return TestFileParser._parse_tsv(file_path)
                else:
                    return TestFileParser._parse_csv(file_path)
    
    @staticmethod
    def _parse_csv(csv_file: str) -> List[Dict]:
        """
        CSVファイルからテストケースを読み込み
        
        Args:
            csv_file: CSVファイルパス
            
        Returns:
            テストケースのリスト
        """
        test_cases = []
        
        try:
            # CSVファイルのエンコーディングを推測
            encodings = ['utf-8', 'shift_jis', 'cp932', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    with open(csv_file, 'r', encoding=encoding) as f:
                        reader = csv.DictReader(f)
                        test_cases = list(reader)
                        if test_cases:
                            logger.info(f"Successfully loaded CSV test cases with encoding: {encoding}")
                            break
                except UnicodeDecodeError:
                    continue
            
            if not test_cases:
                logger.error(f"Failed to load test cases from {csv_file}: Could not determine encoding")
                return []
            
            logger.info(f"Loaded {len(test_cases)} test cases from CSV file")
            return test_cases
        except Exception as e:
            logger.error(f"Failed to load test cases from {csv_file}: {e}")
            return []
    
    @staticmethod
    def _parse_tsv(tsv_file: str) -> List[Dict]:
        """
        TSVファイルからテストケースを読み込み
        
        Args:
            tsv_file: TSVファイルパス
            
        Returns:
            テストケースのリスト
        """
        test_cases = []
        
        try:
            # TSVファイルのエンコーディングを推測
            encodings = ['utf-8', 'shift_jis', 'cp932', 'iso-8859-1']
            
            file_content = None
            used_encoding = None
            
            for encoding in encodings:
                try:
                    with open(tsv_file, 'r', encoding=encoding) as f:
                        file_content = f.read()
                        used_encoding = encoding
                        logger.info(f"Successfully read TSV file with encoding: {encoding}")
                        break
                except UnicodeDecodeError:
                    continue
            
            if not file_content:
                logger.error(f"Failed to load test cases from {tsv_file}: Could not determine encoding")
                return []
            
            # ヘッダー行と内容行に分割
            lines = file_content.split('\n')
            if not lines:
                logger.error(f"TSV file is empty: {tsv_file}")
                return []
            
            # ヘッダー行からフィールド名を取得
            header_line = lines[0]
            field_names = header_line.split('\t')
            
            # ヘッダーのマッピングを作成（標準フィールド名へ変換）
            field_mapping = {}
            standard_fields = {
                'カテゴリ': 'category',
                'テストケースID': 'test_case_id', 
                '目的': 'purpose',
                '手順': 'steps',
                '期待結果': 'expected_result',
                '合否判定': 'pass_criteria',
                '前提条件': 'prerequisite',
                '優先度': 'priority',
                'テスト方法': 'test_method',
                'テスト対象ノード': 'node',
                'テストコマンド': 'command',
                '検証方法': 'validation_method'
            }
            
            for i, field in enumerate(field_names):
                # 改行コードなどを削除
                clean_field = field.strip().rstrip('\r')
                
                # 標準フィールド名に変換
                if clean_field in standard_fields:
                    field_mapping[i] = standard_fields[clean_field]
                else:
                    # 不明なフィールドの場合はそのまま使用
                    field_mapping[i] = clean_field
            
            # 各行をパース
            for i in range(1, len(lines)):
                line = lines[i]
                if not line.strip():
                    continue  # 空行はスキップ
                
                fields = line.split('\t')
                test_case = {}
                
                # フィールドを辞書に格納
                for j, field_value in enumerate(fields):
                    if j in field_mapping:
                        # 改行コードなどを保持したままフィールドを格納
                        test_case[field_mapping[j]] = field_value
                
                test_cases.append(test_case)
            
            logger.info(f"Loaded {len(test_cases)} test cases from TSV file")
            return test_cases
        except Exception as e:
            logger.error(f"Failed to load test cases from {tsv_file}: {e}")
            return []
        
class ValidationHandler:
    """テスト検証ハンドラクラス"""
    
    @staticmethod
    def validate_output(output: str, validation_method: str) -> Tuple[bool, str]:
        """
        出力結果を検証
        
        Args:
            output: 検証対象の出力文字列
            validation_method: 検証方法（CONTAINS、REGEX、JSON_CONTAINSなど）
            
        Returns:
            タプル (検証結果, メッセージ)
        """
        if not validation_method or validation_method == 'N/A':
            # 検証方法が指定されていない場合は成功とみなす
            return True, "検証方法が指定されていません"
        
        # 変数抽出のための特殊構文を無視する
        if 'JSON_EXTRACT(' in validation_method:
            # JSON_EXTRACT部分を除去して検証
            clean_validation = re.sub(r'JSON_EXTRACT\([^)]+\)', '', validation_method).strip()
            if not clean_validation:
                return True, "JSON_EXTRACT のみの指定のため、検証はスキップします"
            validation_method = clean_validation
        
        # 複数の検証条件（AND, OR）処理
        if '&&' in validation_method:
            conditions = validation_method.split('&&')
            for condition in conditions:
                result, message = ValidationHandler.validate_single_condition(output, condition.strip())
                if not result:
                    return False, f"AND条件が不成立: {message}"
            return True, "全てのAND条件が成立しました"
        
        elif '||' in validation_method:
            conditions = validation_method.split('||')
            for condition in conditions:
                result, message = ValidationHandler.validate_single_condition(output, condition.strip())
                if result:
                    return True, f"OR条件が成立: {message}"
            return False, "いずれのOR条件も成立しませんでした"
        
        # 単一条件の検証
        return ValidationHandler.validate_single_condition(output, validation_method)
    
    @staticmethod
    def validate_single_condition(output: str, condition: str) -> Tuple[bool, str]:
        """
        単一の条件に対する検証
        
        Args:
            output: 検証対象の出力文字列
            condition: 検証条件（CONTAINS(xxx)形式）
            
        Returns:
            タプル (検証結果, メッセージ)
        """
        # 否定条件の処理
        is_negated = False
        if condition.startswith('!'):
            is_negated = True
            condition = condition[1:]
        
        # CONTAINS
        if condition.startswith('CONTAINS(') and condition.endswith(')'):
            search_text = condition[9:-1]  # CONTAINS( と ) を除去
            
            # パイプ記号で区切られた複数の候補を処理
            if '|' in search_text:
                search_texts = [s.strip() for s in search_text.split('|')]
                found = any(text in output for text in search_texts)
                if found ^ is_negated:  # XOR演算子で否定処理
                    texts_str = ' または '.join([f"'{t}'" for t in search_texts])
                    return True, f"出力に{texts_str}が含まれています" if not is_negated else f"出力に{texts_str}が含まれていません"
                else:
                    texts_str = ' または '.join([f"'{t}'" for t in search_texts])
                    return False, f"出力に{texts_str}が含まれていません" if not is_negated else f"出力に{texts_str}が含まれています"
            
            # 単一の検索テキスト
            found = search_text in output
            if found ^ is_negated:  # XOR演算子で否定処理
                return True, f"出力に'{search_text}'が含まれています" if not is_negated else f"出力に'{search_text}'が含まれていません"
            else:
                return False, f"出力に'{search_text}'が含まれていません" if not is_negated else f"出力に'{search_text}'が含まれています"
        
        # REGEX
        elif condition.startswith('REGEX(') and condition.endswith(')'):
            pattern = condition[6:-1]  # REGEX( と ) を除去
            try:
                match = re.search(pattern, output)
                if (match is not None) ^ is_negated:  # XOR演算子で否定処理
                    return True, f"正規表現 '{pattern}' が一致しました" if not is_negated else f"正規表現 '{pattern}' が一致しませんでした"
                else:
                    return False, f"正規表現 '{pattern}' が一致しませんでした" if not is_negated else f"正規表現 '{pattern}' が一致しました"
            except re.error as e:
                return False, f"正規表現エラー: {e}"
        
        # JSON_CONTAINS
        elif condition.startswith('JSON_CONTAINS(') and condition.endswith(')'):
            params_str = condition[13:-1]  # JSON_CONTAINS( と ) を除去
            
            # パラメータを解析
            params = params_str.split(',')
            if len(params) != 2:
                return False, f"JSON_CONTAINS構文エラー: 2つのパラメータが必要です: {params_str}"
            
            key = params[0].strip()
            expected_value = params[1].strip()
            
            try:
                # JSONをパース
                json_data = json.loads(output)
                
                # ネストした階層があるか確認 (data.key のような形式)
                if '.' in key:
                    parts = key.split('.')
                    current = json_data
                    for part in parts:
                        if part in current:
                            current = current[part]
                        else:
                            return False, f"JSONキー '{key}' が見つかりません"
                    
                    # 文字列化して比較（数値や真偽値も対応）
                    actual_value = str(current)
                else:
                    if key not in json_data:
                        return False, f"JSONキー '{key}' が見つかりません"
                    
                    # 文字列化して比較（数値や真偽値も対応）
                    actual_value = str(json_data[key])
                
                # 値を比較
                matches = (actual_value == expected_value)
                if matches ^ is_negated:  # XOR演算子で否定処理
                    return True, f"JSONキー '{key}' の値 '{actual_value}' が期待値 '{expected_value}' と一致しました" if not is_negated else f"JSONキー '{key}' の値 '{actual_value}' が期待値 '{expected_value}' と一致しませんでした"
                else:
                    return False, f"JSONキー '{key}' の値 '{actual_value}' が期待値 '{expected_value}' と一致しませんでした" if not is_negated else f"JSONキー '{key}' の値 '{actual_value}' が期待値 '{expected_value}' と一致しました"
            
            except json.JSONDecodeError:
                return False, "出力がJSON形式ではありません"
            except Exception as e:
                return False, f"JSON検証エラー: {e}"
                
        # JSON_TYPE
        elif condition.startswith('JSON_TYPE(') and condition.endswith(')'):
            params_str = condition[10:-1]  # JSON_TYPE( と ) を除去
            
            # パラメータを解析
            params = params_str.split(',')
            if len(params) != 2:
                return False, f"JSON_TYPE構文エラー: 2つのパラメータが必要です: {params_str}"
            
            key = params[0].strip()
            expected_type = params[1].strip()
            
            try:
                # JSONをパース
                json_data = json.loads(output)
                
                # ネストした階層があるか確認 (data.key のような形式)
                if '.' in key:
                    parts = key.split('.')
                    current = json_data
                    for part in parts:
                        if part in current:
                            current = current[part]
                        else:
                            return False, f"JSONキー '{key}' が見つかりません"
                    
                    # 型を確認
                    actual_type = type(current).__name__
                else:
                    if key not in json_data:
                        return False, f"JSONキー '{key}' が見つかりません"
                    
                    # 型を確認
                    actual_type = type(json_data[key]).__name__
                
                # 型を比較
                matches = (actual_type == expected_type)
                if matches ^ is_negated:  # XOR演算子で否定処理
                    return True, f"JSONキー '{key}' の型 '{actual_type}' が期待値 '{expected_type}' と一致しました" if not is_negated else f"JSONキー '{key}' の型 '{actual_type}' が期待値 '{expected_type}' と一致しませんでした"
                else:
                    return False, f"JSONキー '{key}' の型 '{actual_type}' が期待値 '{expected_type}' と一致しませんでした" if not is_negated else f"JSONキー '{key}' の型 '{actual_type}' が期待値 '{expected_type}' と一致しました"
            
            except json.JSONDecodeError:
                return False, "出力がJSON形式ではありません"
            except Exception as e:
                return False, f"JSON型検証エラー: {e}"
        
        # 不明な検証方法
        else:
            return False, f"不明な検証方法: {condition}"

class TestExecutor:
    """テスト実行クラス"""
    
    def __init__(self, proxmox_tester):
        """
        初期化メソッド
        
        Args:
            proxmox_tester: ProxmoxTesterインスタンス
        """
        self.tester = proxmox_tester
        self.logger = logging.getLogger("TestExecutor")
        # テスト実行中のコンテキスト情報を保持
        self.context = {}
    
    def execute_test(self, test_case: Dict) -> TestResult:
        """
        テストを実行
        
        Args:
            test_case: テストケース辞書
            
        Returns:
            テスト結果
        """
        test_id = test_case.get('test_case_id', str(uuid.uuid4())[:8])
        category = test_case.get('category', 'uncategorized')
        test_name = test_case.get('purpose', test_id)
        test_method = test_case.get('test_method', '')
        node_key = test_case.get('node', '')
        command = test_case.get('command', '')
        validation_method = test_case.get('validation_method', '')
        
        # 結果オブジェクトを初期化
        result = TestResult(test_id, category, test_name)
        
        # 実行コンテキストをリセット
        self.context = {
            'test_id': test_id,
            'node': node_key,
            'variables': {}
        }
        
        self.logger.info(f"Running test case {test_id}: {test_name} (Method: {test_method})")
        result.add_detail(f"=== テストケース {test_id}: {test_name} ===")
        result.add_detail(f"カテゴリ: {category}")
        result.add_detail(f"テスト方法: {test_method}")
        result.add_detail(f"テスト対象ノード: {node_key}")
        
        # 手順と期待結果を記録
        if 'steps' in test_case and test_case['steps']:
            result.add_detail(f"手順:\n{test_case['steps']}")
        
        if 'expected_result' in test_case and test_case['expected_result']:
            result.add_detail(f"期待結果:\n{test_case['expected_result']}")
        
        # 手動テストはスキップ
        if test_method.lower() == '手動':
            result.add_detail("手動テストのため自動実行はスキップされました")
            result.success = None  # スキップ
            return result
        
        try:
            # テスト方法に応じた実行
            if test_method.lower() == 'ssh':
                return self._execute_ssh_test(test_case, result)
            elif test_method.lower() == 'api':
                return self._execute_api_test(test_case, result)
            elif test_method.lower() == 'api+ssh':
                return self._execute_api_ssh_test(test_case, result)
            else:
                result.set_failure(f"不明なテスト方法: {test_method}")
                return result
        
        except Exception as e:
            self.logger.exception(f"テスト実行中にエラーが発生: {str(e)}")
            result.set_failure(f"テスト実行エラー: {str(e)}", e)
            return result
    
    def _expand_variables(self, command: str) -> str:
        """
        コマンド文字列内の変数を展開する
        
        Args:
            command: 変数を含む可能性のあるコマンド文字列
            
        Returns:
            変数が展開されたコマンド文字列
        """
        # {vmid} のような変数参照を context から置換
        for var_name, var_value in self.context.get('variables', {}).items():
            placeholder = "{" + var_name + "}"
            if placeholder in command:
                command = command.replace(placeholder, str(var_value))
        
        return command
    
    def _extract_variables(self, output: str, validation_method: str) -> Dict:
        """
        出力から変数を抽出する
        例: JSON_EXTRACT(data.vmid, vmid) は data.vmid の値を vmid 変数に保存
        
        Args:
            output: テスト出力
            validation_method: 検証/抽出メソッド
            
        Returns:
            抽出された変数の辞書
        """
        variables = {}
        
        # JSON_EXTRACT構文を解析
        if validation_method and 'JSON_EXTRACT(' in validation_method:
            extract_parts = validation_method.split('JSON_EXTRACT(')
            for part in extract_parts[1:]:  # 最初の部分はスキップ
                if ')' not in part:
                    continue
                    
                extract_expr = part.split(')', 1)[0]  # 最初の閉じ括弧までを取得
                if ',' not in extract_expr:
                    continue
                    
                json_path, var_name = [p.strip() for p in extract_expr.split(',', 1)]
                
                try:
                    # JSONをパース
                    json_data = json.loads(output)
                    
                    # ネストした階層があるか確認 (data.key のような形式)
                    if '.' in json_path:
                        parts = json_path.split('.')
                        current = json_data
                        for path_part in parts:
                            if path_part in current:
                                current = current[path_part]
                            else:
                                self.logger.warning(f"JSONパス '{json_path}' が見つかりません")
                                break
                        else:
                            # 変数に値を保存
                            variables[var_name] = current
                    else:
                        if json_path in json_data:
                            # 変数に値を保存
                            variables[var_name] = json_data[json_path]
                
                except json.JSONDecodeError:
                    self.logger.warning("JSONの解析に失敗しました")
                except Exception as e:
                    self.logger.warning(f"変数抽出中にエラーが発生: {e}")
        
        # REGEXキャプチャ構文を解析 (未実装の例)
        # REGEX_CAPTURE(pattern, group_name, var_name)
        
        return variables
    
    def _execute_ssh_test(self, test_case: Dict, result: TestResult) -> TestResult:
        """
        SSHテストを実行
        
        Args:
            test_case: テストケース辞書
            result: テスト結果
            
        Returns:
            更新されたテスト結果
        """
        node_key = test_case.get('node', '')
        command = test_case.get('command', '')
        validation_method = test_case.get('validation_method', '')
        
        # ノード情報を取得
        node_info = self.tester.nodes.get(node_key, None)
        if not node_info:
            err_msg = f"不明なノード: {node_key}"
            self.logger.error(err_msg)
            result.set_failure(err_msg)
            return result
        
        # ホスト名を取得
        host = node_info.get('host', '')
        if not host:
            err_msg = f"ノード {node_key} のホスト名が設定されていません"
            self.logger.error(err_msg)
            result.set_failure(err_msg)
            return result
        
        # コマンドがN/Aの場合はスキップ
        if not command or command.upper() == 'N/A':
            result.add_detail("コマンドが指定されていないため、スキップします")
            result.success = None  # スキップ
            return result
        
        # コマンド内の変数を展開
        command = self._expand_variables(command)
        
        # SSHクライアントを初期化
        ssh_client = self.tester.get_ssh_client(host)
        
        try:
            # SSH接続
            if not ssh_client.connect():
                err_msg = f"SSH接続に失敗: {host}"
                self.logger.error(err_msg)
                result.set_failure(err_msg)
                return result
            
            # コマンド実行
            result.add_detail(f"コマンド実行: {command}")
            exit_code, stdout, stderr = ssh_client.execute_command(command)
            
            # 結果を記録
            result.add_detail(f"終了コード: {exit_code}")
            result.add_detail(f"標準出力:\n{stdout}")
            
            if stderr:
                result.add_detail(f"標準エラー出力:\n{stderr}")
            
            # 出力から変数を抽出
            if validation_method:
                extracted_vars = self._extract_variables(stdout, validation_method)
                if extracted_vars:
                    self.context['variables'].update(extracted_vars)
                    result.add_detail(f"抽出された変数: {extracted_vars}")
            
            # 検証
            if validation_method:
                result.add_detail(f"検証方法: {validation_method}")
                is_valid, validation_message = ValidationHandler.validate_output(stdout, validation_method)
                result.add_detail(f"検証結果: {validation_message}")
                
                if is_valid:
                    result.set_success(f"テスト成功: {validation_message}")
                else:
                    result.set_failure(f"テスト失敗: {validation_message}")
            else:
                # 検証方法がない場合は終了コードで判定
                if exit_code == 0:
                    result.set_success("テスト成功: 終了コード 0")
                else:
                    result.set_failure(f"テスト失敗: 終了コード {exit_code}")
            
            return result
        
        finally:
            # SSH接続を閉じる
            ssh_client.close()
    
    def _execute_api_test(self, test_case: Dict, result: TestResult) -> TestResult:
        """
        APIテストを実行
        
        Args:
            test_case: テストケース辞書
            result: テスト結果
            
        Returns:
            更新されたテスト結果
        """
        command = test_case.get('command', '')
        validation_method = test_case.get('validation_method', '')
        
        # コマンドがN/Aの場合はスキップ
        if not command or command.upper() == 'N/A':
            result.add_detail("APIコマンドが指定されていないため、スキップします")
            result.success = None  # スキップ
            return result
        
        # コマンド内の変数を展開
        command = self._expand_variables(command)
        
        # APIコマンドをパース
        command_parts = command.strip().split()
        if not command_parts:
            result.set_failure("APIコマンドが空です")
            return result
        
        http_method = command_parts[0].upper()
        endpoint = None
        data = {}
        
        if len(command_parts) > 1:
            endpoint = command_parts[1]
        
        if len(command_parts) > 2:
            # データパラメータを解析（key=value形式）
            for param in command_parts[2:]:
                if '=' in param:
                    key, value = param.split('=', 1)
                    data[key] = value
        
        if not endpoint:
            result.set_failure("APIエンドポイントが指定されていません")
            return result
        
        # APIリクエストを実行
        result.add_detail(f"APIリクエスト: {http_method} {endpoint}")
        if data:
            result.add_detail(f"パラメータ: {data}")
        
        try:
            response = None
            
            if http_method == 'GET':
                response = self.tester.api_client.get(endpoint)
            elif http_method == 'POST':
                response = self.tester.api_client.post(endpoint, data)
            elif http_method == 'PUT':
                response = self.tester.api_client.put(endpoint, data)
            elif http_method == 'DELETE':
                response = self.tester.api_client.delete(endpoint, data)
            else:
                result.set_failure(f"サポートされていないHTTPメソッド: {http_method}")
                return result
            
            # 応答を記録
            response_str = json.dumps(response, indent=2)
            result.add_detail(f"API応答:\n{response_str}")
            
            # レスポンスデータから変数を抽出して保存
            if validation_method and response:
                extracted_vars = self._extract_variables(response_str, validation_method)
                if extracted_vars:
                    self.context['variables'].update(extracted_vars)
                    result.add_detail(f"抽出された変数: {extracted_vars}")
            
            # 検証
            if validation_method:
                result.add_detail(f"検証方法: {validation_method}")
                is_valid, validation_message = ValidationHandler.validate_output(response_str, validation_method)
                result.add_detail(f"検証結果: {validation_message}")
                
                if is_valid:
                    result.set_success(f"テスト成功: {validation_message}")
                else:
                    result.set_failure(f"テスト失敗: {validation_message}")
            else:
                # 検証方法がない場合は応答にエラーキーがあるかどうかで判定
                if isinstance(response, dict) and "error" in response:
                    result.set_failure(f"テスト失敗: APIエラー {response.get('error')}")
                else:
                    result.set_success("テスト成功: APIエラーなし")
            
            return result
        
        except Exception as e:
            result.set_failure(f"APIリクエスト実行エラー: {str(e)}", e)
            return result
    
    def _execute_api_ssh_test(self, test_case: Dict, result: TestResult) -> TestResult:
        """
        API+SSHテストを実行（APIとSSHを組み合わせた統合テスト）
        
        Args:
            test_case: テストケース辞書
            result: テスト結果
            
        Returns:
            更新されたテスト結果
        """
        # テストコマンドを解析
        command = test_case.get('command', '')
        
        # API部分とSSH部分に分割するケース（";" で区切られている場合）
        if ';' in command:
            api_command, ssh_command = command.split(';', 1)
            
            # 一時的にコマンドを置き換えてAPIテストを実行
            api_test_case = test_case.copy()
            api_test_case['command'] = api_command.strip()
            api_test_case['test_method'] = 'api'
            
            # APIテストを実行
            api_result = self._execute_api_test(api_test_case, result)
            
            # APIテストが失敗した場合は終了
            if api_result.success is False:
                return api_result
                
            # APIの結果を使ってSSHテストを実行
            ssh_test_case = test_case.copy()
            ssh_test_case['command'] = ssh_command.strip()
            ssh_test_case['test_method'] = 'ssh'
            
            return self._execute_ssh_test(ssh_test_case, result)
            
        # または、API実行後にレスポンスをもとにSSHコマンドを生成するケース
        elif command.startswith("API+SSH:"):
            # API+SSH:GET /path/to/api:echo {response.value} > /tmp/test
            parts = command[8:].strip().split(':', 1)
            if len(parts) != 2:
                result.set_failure("無効なAPI+SSHコマンド形式")
                return result
                
            api_command, ssh_template = parts
            
            # API実行
            api_test_case = test_case.copy()
            api_test_case['command'] = api_command.strip()
            api_test_case['test_method'] = 'api'
            
            api_result = self._execute_api_test(api_test_case, result)
            
            # APIテストが失敗した場合は終了
            if api_result.success is False:
                return api_result
                
            # SSH実行
            ssh_command = self._expand_variables(ssh_template.strip())
            ssh_test_case = test_case.copy()
            ssh_test_case['command'] = ssh_command
            ssh_test_case['test_method'] = 'ssh'
            
            return self._execute_ssh_test(ssh_test_case, result)
            
        # デフォルトの実装（単にAPI実行後にSSH実行）
        else:
            # まずAPIテストを実行
            api_test_case = test_case.copy()
            api_test_case['test_method'] = 'api'
            api_result = self._execute_api_test(api_test_case, result)
            
            # APIテストが失敗した場合は終了
            if api_result.success is False:
                return api_result
            
            # 次にSSHテストを実行
            ssh_test_case = test_case.copy()
            ssh_test_case['test_method'] = 'ssh'
            return self._execute_ssh_test(ssh_test_case, result)
        
class ProxmoxAPIClient:
    """Proxmox VE API クライアントクラス"""
    
    def __init__(self, host: str, token_id: str, token_secret: str, port: int = 8006, verify_ssl: bool = False):
        """
        初期化メソッド
        
        Args:
            host: Proxmoxホスト名またはIPアドレス
            token_id: APIトークンID（例: 'root@pam!apitoken'）
            token_secret: APIトークンシークレット
            port: APIポート（デフォルト: 8006）
            verify_ssl: SSL証明書を検証するかどうか（デフォルト: False）
        """
        self.base_url = f"https://{host}:{port}/api2/json"
        self.headers = {
            "Authorization": f"PVEAPIToken={token_id}={token_secret}"
        }
        self.verify_ssl = verify_ssl
        self.session = requests.Session()
        logger.info(f"ProxmoxAPIClient initialized for host: {host}")
    
    def get(self, endpoint: str) -> Dict:
        """
        GETリクエストを送信
        
        Args:
            endpoint: APIエンドポイント（/から始まる相対パス）
            
        Returns:
            レスポンスデータ（JSONをパースしたディクショナリ）
        """
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.get(url, headers=self.headers, verify=self.verify_ssl)
            response.raise_for_status()
            return response.json().get('data', {})
        except Exception as e:
            logger.error(f"GET request failed: {url}, error: {e}")
            return {"error": str(e)}
    
    def post(self, endpoint: str, data: Dict = None) -> Dict:
        """
        POSTリクエストを送信
        
        Args:
            endpoint: APIエンドポイント（/から始まる相対パス）
            data: POSTするデータ（省略可）
            
        Returns:
            レスポンスデータ（JSONをパースしたディクショナリ）
        """
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.post(url, headers=self.headers, data=data, verify=self.verify_ssl)
            response.raise_for_status()
            return response.json().get('data', {})
        except Exception as e:
            logger.error(f"POST request failed: {url}, error: {e}")
            return {"error": str(e)}
    
    def put(self, endpoint: str, data: Dict = None) -> Dict:
        """
        PUTリクエストを送信
        
        Args:
            endpoint: APIエンドポイント（/から始まる相対パス）
            data: PUTするデータ（省略可）
            
        Returns:
            レスポンスデータ（JSONをパースしたディクショナリ）
        """
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.put(url, headers=self.headers, data=data, verify=self.verify_ssl)
            response.raise_for_status()
            return response.json().get('data', {})
        except Exception as e:
            logger.error(f"PUT request failed: {url}, error: {e}")
            return {"error": str(e)}
    
    def delete(self, endpoint: str, data: Dict = None) -> Dict:
        """
        DELETEリクエストを送信
        
        Args:
            endpoint: APIエンドポイント（/から始まる相対パス）
            data: DELETEするデータ（省略可）
            
        Returns:
            レスポンスデータ（JSONをパースしたディクショナリ）
        """
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.delete(url, headers=self.headers, data=data, verify=self.verify_ssl)
            response.raise_for_status()
            return response.json().get('data', {})
        except Exception as e:
            logger.error(f"DELETE request failed: {url}, error: {e}")
            return {"error": str(e)}
    
    def get_nodes(self) -> List[Dict]:
        """クラスター内のノード一覧を取得"""
        return self.get("/nodes")
    
    def get_node_status(self, node: str) -> Dict:
        """指定ノードのステータスを取得"""
        return self.get(f"/nodes/{node}/status")
    
    def get_cluster_status(self) -> Dict:
        """クラスターの状態を取得"""
        return self.get("/cluster/status")
    
    def get_cluster_resources(self) -> List[Dict]:
        """クラスターリソースを取得"""
        return self.get("/cluster/resources")
    
    def get_ceph_status(self, node: str) -> Dict:
        """Cephステータスを取得"""
        return self.get(f"/nodes/{node}/ceph/status")
    
    def get_vms(self, node: str) -> List[Dict]:
        """指定ノードのVM一覧を取得"""
        return self.get(f"/nodes/{node}/qemu")
    
    def get_vm_status(self, node: str, vmid: int) -> Dict:
        """指定VMのステータスを取得"""
        return self.get(f"/nodes/{node}/qemu/{vmid}/status/current")
    
    def create_vm(self, node: str, vm_params: Dict) -> Dict:
        """VMを作成"""
        return self.post(f"/nodes/{node}/qemu", vm_params)
    
    def start_vm(self, node: str, vmid: int) -> Dict:
        """VMを起動"""
        return self.post(f"/nodes/{node}/qemu/{vmid}/status/start")
    
    def stop_vm(self, node: str, vmid: int) -> Dict:
        """VMを停止"""
        return self.post(f"/nodes/{node}/qemu/{vmid}/status/stop")
    
    def shutdown_vm(self, node: str, vmid: int) -> Dict:
        """VMをシャットダウン"""
        return self.post(f"/nodes/{node}/qemu/{vmid}/status/shutdown")
    
    def create_snapshot(self, node: str, vmid: int, snapshot_name: str, description: str = "") -> Dict:
        """VMのスナップショットを作成"""
        data = {
            "snapname": snapshot_name,
            "description": description
        }
        return self.post(f"/nodes/{node}/qemu/{vmid}/snapshot", data)
    
    def rollback_snapshot(self, node: str, vmid: int, snapshot_name: str) -> Dict:
        """VMをスナップショットに戻す"""
        return self.post(f"/nodes/{node}/qemu/{vmid}/snapshot/{snapshot_name}/rollback")
    
    def get_ha_status(self) -> List[Dict]:
        """HAリソースの状態を取得"""
        return self.get("/cluster/ha/status")
    
    def get_ha_resources(self) -> List[Dict]:
        """HAリソース一覧を取得"""
        return self.get("/cluster/ha/resources")

class SSHClient:
    """SSHクライアントクラス"""
    
    def __init__(self, host: str, username: str, password: str = None, key_filename: str = None, port: int = 22):
        """
        初期化メソッド
        
        Args:
            host: ホスト名またはIPアドレス
            username: SSHユーザー名
            password: パスワード認証の場合のパスワード（省略可）
            key_filename: 鍵認証の場合の秘密鍵ファイルパス（省略可）
            port: SSHポート番号（デフォルト: 22）
        """
        self.host = host
        self.username = username
        self.password = password
        self.key_filename = key_filename
        self.port = port
        self.client = None
        logger.info(f"SSHClient initialized for host: {host}")
    
    def connect(self) -> bool:
        """
        SSH接続を確立
        
        Returns:
            接続成功の場合はTrue、失敗の場合はFalse
        """
        try:
            self.client = paramiko.SSHClient()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            connect_args = {
                "hostname": self.host,
                "username": self.username,
                "port": self.port,
                "timeout": 10
            }
            
            if self.password:
                connect_args["password"] = self.password
            if self.key_filename:
                connect_args["key_filename"] = self.key_filename
                
            self.client.connect(**connect_args)
            logger.info(f"SSH connection established to {self.host}")
            return True
        except Exception as e:
            logger.error(f"SSH connection failed to {self.host}: {e}")
            return False
    
    def execute_command(self, command: str, timeout: int = 60) -> Tuple[int, str, str]:
        """
        コマンド実行
        
        Args:
            command: 実行するコマンド
            timeout: コマンド実行タイムアウト（秒）
            
        Returns:
            タプル (終了コード, 標準出力, 標準エラー出力)
        """
        if not self.client:
            logger.error("SSH connection not established")
            return (-1, "", "SSH connection not established")
        
        try:
            stdin, stdout, stderr = self.client.exec_command(command, timeout=timeout)
            exit_code = stdout.channel.recv_exit_status()
            stdout_str = stdout.read().decode('utf-8')
            stderr_str = stderr.read().decode('utf-8')
            
            logger.info(f"Command executed: '{command}', exit code: {exit_code}")
            if stderr_str:
                logger.debug(f"stderr: {stderr_str}")
            
            return (exit_code, stdout_str, stderr_str)
        except Exception as e:
            logger.error(f"Command execution failed: '{command}', error: {e}")
            return (-1, "", str(e))
    
    def close(self) -> None:
        """SSH接続を閉じる"""
        if self.client:
            self.client.close()
            self.client = None
            logger.info(f"SSH connection closed to {self.host}")


class NetworkTools:
    """ネットワークツールクラス"""
    
    @staticmethod
    def ping(host: str, count: int = 5, timeout: int = 5, interface: str = None) -> Tuple[bool, str, float]:
        """
        指定ホストにpingを実行
        
        Args:
            host: pingを送信するホスト
            count: pingの回数
            timeout: タイムアウト（秒）
            interface: 送信元インターフェース（省略可）
            
        Returns:
            タプル (成功/失敗, 出力メッセージ, パケットロス率)
        """
        try:
            cmd = ['ping']
            
            if sys.platform == 'win32':
                cmd.extend(['-n', str(count), '-w', str(timeout * 1000)])
                if interface:
                    source_ip = NetworkTools.get_interface_ip(interface)
                    if source_ip:
                        cmd.extend(['-S', source_ip])
            else:
                cmd.extend(['-c', str(count), '-W', str(timeout)])
                if interface:
                    cmd.extend(['-I', interface])
            
            cmd.append(host)
            
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            stdout, stderr = process.communicate(timeout=timeout * count + 5)
            
            # パケットロス率を計算
            packet_loss = 100.0  # デフォルトは100%（全ロス）
            
            if sys.platform == 'win32':
                match = re.search(r'(\d+)% loss', stdout)
                if match:
                    packet_loss = float(match.group(1))
            else:
                match = re.search(r'(\d+)% packet loss', stdout)
                if match:
                    packet_loss = float(match.group(1))
            
            result = process.returncode == 0
            return (result, stdout, packet_loss)
        except subprocess.TimeoutExpired:
            return (False, "Ping command timed out", 100.0)
        except Exception as e:
            return (False, str(e), 100.0)
    
    @staticmethod
    def get_interface_ip(interface_name: str) -> Optional[str]:
        """
        指定されたネットワークインターフェースのIPアドレスを取得
        
        Args:
            interface_name: ネットワークインターフェース名
            
        Returns:
            IPv4アドレス（見つからない場合はNone）
        """
        try:
            for interface, addrs in psutil.net_if_addrs().items():
                if interface == interface_name:
                    for addr in addrs:
                        if addr.family == socket.AF_INET:
                            return addr.address
        except Exception as e:
            logger.error(f"Error getting IP for interface {interface_name}: {e}")
        
        return None
    
    @staticmethod
    def check_web_access(url: str, expected_status: int = 200, expected_content: str = None, timeout: int = 10) -> Tuple[bool, str]:
        """
        Web URLへのアクセスをチェック
        
        Args:
            url: アクセスするURL
            expected_status: 期待するHTTPステータスコード
            expected_content: レスポンス内に期待する文字列（省略可）
            timeout: タイムアウト（秒）
            
        Returns:
            タプル (成功/失敗, メッセージ)
        """
        try:
            response = requests.get(url, timeout=timeout, verify=False)
            
            if response.status_code != expected_status:
                return (False, f"HTTPステータスコード不一致: 期待値={expected_status}, 実際={response.status_code}")
            
            if expected_content and expected_content not in response.text:
                return (False, f"期待するコンテンツが見つかりません: {expected_content}")
            
            return (True, f"アクセス成功: {url}")
        except requests.exceptions.Timeout:
            return (False, f"タイムアウト: {url}")
        except requests.exceptions.ConnectionError:
            return (False, f"接続エラー: {url}")
        except Exception as e:
            return (False, f"エラー: {str(e)}")

def generate_report(self, output_file: str) -> None:
    """
    テスト結果のレポートをMarkdown形式で生成
    
    Args:
        output_file: 出力ファイルパス
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # レポートヘッダー
            f.write("# Proxmoxクラスターテスト結果レポート\n\n")
            f.write(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # サマリー
            success_count = sum(1 for r in self.results if r.success is True)
            failure_count = sum(1 for r in self.results if r.success is False)
            skipped_count = sum(1 for r in self.results if r.success is None)
            total_count = len(self.results)
            
            f.write("## テスト実行サマリー\n\n")
            f.write("| 項目 | 結果 |\n")
            f.write("|------|------|\n")
            f.write(f"| 合格 | {success_count} |\n")
            f.write(f"| 不合格 | {failure_count} |\n")
            f.write(f"| スキップ | {skipped_count} |\n")
            f.write(f"| 合計 | {total_count} |\n\n")
            
            # カテゴリ別結果
            f.write("## カテゴリ別結果\n\n")
            categories = {}
            for result in self.results:
                if result.category not in categories:
                    categories[result.category] = {
                        'success': 0,
                        'failure': 0,
                        'skipped': 0,
                        'total': 0
                    }
                
                categories[result.category]['total'] += 1
                if result.success is True:
                    categories[result.category]['success'] += 1
                elif result.success is False:
                    categories[result.category]['failure'] += 1
                else:
                    categories[result.category]['skipped'] += 1
            
            f.write("| カテゴリ | 合格 | 不合格 | スキップ | 合計 |\n")
            f.write("|----------|------|--------|----------|------|\n")
            
            for category, stats in sorted(categories.items()):
                f.write(f"| {category} | {stats['success']} | {stats['failure']} | {stats['skipped']} | {stats['total']} |\n")
            
            f.write("\n")
            
            # 詳細結果
            f.write("## テスト詳細結果\n\n")
            
            for result in self.results:
                status = "⭕ 合格" if result.success is True else "❌ 不合格" if result.success is False else "⏩ スキップ"
                f.write(f"### {result.test_id}: {result.name} ({status})\n\n")
                f.write(f"- **カテゴリ**: {result.category}\n")
                f.write(f"- **実行時間**: {result.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"- **所要時間**: {result.duration:.2f}秒\n")
                f.write(f"- **結果**: {result.message}\n\n")
                
                f.write("#### 詳細\n\n")
                for detail in result.details:
                    f.write(f"{detail}\n\n")
                
                if result.error:
                    f.write("#### エラー詳細\n\n")
                    f.write(f"```\n{str(result.error)}\n```\n\n")
                
                f.write("---\n\n")
            
            logger.info(f"テスト結果レポートを生成しました: {output_file}")
    except Exception as e:
        logger.error(f"レポート生成中にエラーが発生: {e}")

def generate_junit_report(self, output_file: str) -> None:
    """
    テスト結果のレポートをJUnit XML形式で生成
    
    Args:
        output_file: 出力ファイルパス
    """
    try:
        import xml.etree.ElementTree as ET
        from xml.dom import minidom
        
        # JUnitXML形式のルート要素
        test_suites = ET.Element("testsuites")
        
        # カテゴリごとにテストスイートを作成
        categories = {}
        for result in self.results:
            if result.category not in categories:
                categories[result.category] = []
            categories[result.category].append(result)
        
        for category, results in sorted(categories.items()):
            test_suite = ET.SubElement(test_suites, "testsuite")
            test_suite.set("name", category)
            test_suite.set("tests", str(len(results)))
            test_suite.set("failures", str(sum(1 for r in results if r.success is False)))
            test_suite.set("skipped", str(sum(1 for r in results if r.success is None)))
            test_suite.set("timestamp", datetime.now().isoformat())
            
            for result in results:
                test_case = ET.SubElement(test_suite, "testcase")
                test_case.set("name", result.name)
                test_case.set("classname", result.test_id)
                test_case.set("time", str(result.duration))
                
                if result.success is False:
                    failure = ET.SubElement(test_case, "failure")
                    failure.set("message", result.message)
                    failure.text = "\n".join(result.details)
                    if result.error:
                        failure.text += f"\n\nError: {str(result.error)}"
                
                elif result.success is None:
                    skipped = ET.SubElement(test_case, "skipped")
                    skipped.set("message", result.message or "テストがスキップされました")
                
                system_out = ET.SubElement(test_case, "system-out")
                system_out.text = "\n".join(result.details)
        
        # XMLを整形して出力
        rough_string = ET.tostring(test_suites, "utf-8")
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(pretty_xml)
        
        logger.info(f"JUnit XMLレポートを生成しました: {output_file}")
    except Exception as e:
        logger.error(f"JUnit XMLレポート生成中にエラーが発生: {e}")

class TestResult:
    """テスト結果クラス"""
    
    def __init__(self, test_id: str, category: str, name: str, start_time: datetime = None):
        """
        初期化メソッド
        
        Args:
            test_id: テストケースID
            category: テストカテゴリ
            name: テスト名
            start_time: 開始時刻（省略時は現在時刻）
        """
        self.test_id = test_id
        self.category = category
        self.name = name
        self.start_time = start_time or datetime.now()
        self.end_time = None
        self.success = None  # None: 未実行, True: 成功, False: 失敗
        self.message = ""
        self.details = []
        self.error = None
    
    def set_success(self, message: str = "") -> None:
        """テスト成功を設定"""
        self.success = True
        self.message = message or "テスト成功"
        self.end_time = datetime.now()
    
    def set_failure(self, message: str = "", error: Exception = None) -> None:
        """テスト失敗を設定"""
        self.success = False
        self.message = message or "テスト失敗"
        self.error = error
        self.end_time = datetime.now()
    
    def add_detail(self, message: str) -> None:
        """詳細メッセージを追加"""
        self.details.append(message)
    
    @property
    def duration(self) -> float:
        """テスト実行時間（秒）"""
        if not self.end_time:
            return 0
        delta = self.end_time - self.start_time
        return delta.total_seconds()

def generate_report(self, output_file: str) -> None:
    """
    テスト結果のレポートをMarkdown形式で生成
    
    Args:
        output_file: 出力ファイルパス
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # レポートヘッダー
            f.write("# Proxmoxクラスターテスト結果レポート\n\n")
            f.write(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # サマリー
            success_count = sum(1 for r in self.results if r.success is True)
            failure_count = sum(1 for r in self.results if r.success is False)
            skipped_count = sum(1 for r in self.results if r.success is None)
            total_count = len(self.results)
            
            f.write("## テスト実行サマリー\n\n")
            f.write("| 項目 | 結果 |\n")
            f.write("|------|------|\n")
            f.write(f"| 合格 | {success_count} |\n")
            f.write(f"| 不合格 | {failure_count} |\n")
            f.write(f"| スキップ | {skipped_count} |\n")
            f.write(f"| 合計 | {total_count} |\n\n")
            
            # カテゴリ別結果
            f.write("## カテゴリ別結果\n\n")
            categories = {}
            for result in self.results:
                if result.category not in categories:
                    categories[result.category] = {
                        'success': 0,
                        'failure': 0,
                        'skipped': 0,
                        'total': 0
                    }
                
                categories[result.category]['total'] += 1
                if result.success is True:
                    categories[result.category]['success'] += 1
                elif result.success is False:
                    categories[result.category]['failure'] += 1
                else:
                    categories[result.category]['skipped'] += 1
            
            f.write("| カテゴリ | 合格 | 不合格 | スキップ | 合計 |\n")
            f.write("|----------|------|--------|----------|------|\n")
            
            for category, stats in sorted(categories.items()):
                f.write(f"| {category} | {stats['success']} | {stats['failure']} | {stats['skipped']} | {stats['total']} |\n")
            
            f.write("\n")
            
            # 詳細結果
            f.write("## テスト詳細結果\n\n")
            
            for result in self.results:
                status = "⭕ 合格" if result.success is True else "❌ 不合格" if result.success is False else "⏩ スキップ"
                f.write(f"### {result.test_id}: {result.name} ({status})\n\n")
                f.write(f"- **カテゴリ**: {result.category}\n")
                f.write(f"- **実行時間**: {result.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"- **所要時間**: {result.duration:.2f}秒\n")
                f.write(f"- **結果**: {result.message}\n\n")
                
                f.write("#### 詳細\n\n")
                for detail in result.details:
                    f.write(f"{detail}\n\n")
                
                if result.error:
                    f.write("#### エラー詳細\n\n")
                    f.write(f"```\n{str(result.error)}\n```\n\n")
                
                f.write("---\n\n")
            
            logger.info(f"テスト結果レポートを生成しました: {output_file}")
    except Exception as e:
        logger.error(f"レポート生成中にエラーが発生: {e}")

class ProxmoxTester:
    """Proxmoxテスト自動化のメインクラス"""
    
    def __init__(self, config_file=None):
        """
        初期化メソッド
        
        Args:
            config_file: 設定ファイルパス（省略時は環境変数から設定を読み込み）
        """
        self.logger = logging.getLogger("ProxmoxTester")
        
        # 設定を読み込み
        self.config = self._load_config(config_file)
        
        # APIクライアントを初期化
        self.api_client = self._init_api_client()
        
        # ノード情報を初期化
        self.nodes = self._init_nodes()
        
        # SSHクライアントキャッシュ
        self.ssh_clients = {}
        
        # テスト結果
        self.results = []
        
        # テスト依存関係の追跡
        self.test_dependencies = {}
        
        self.logger.info("ProxmoxTester initialized")

    # 既存の他のメソッドはここでは省略
    
    def generate_report(self, output_file):
        """テスト結果のレポートをMarkdown形式で生成"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                # レポートヘッダー
                f.write("# Proxmoxクラスターテスト結果レポート\n\n")
                f.write(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # サマリー
                success_count = sum(1 for r in self.results if r.success is True)
                failure_count = sum(1 for r in self.results if r.success is False)
                skipped_count = sum(1 for r in self.results if r.success is None)
                total_count = len(self.results)
                
                f.write("## テスト実行サマリー\n\n")
                f.write("| 項目 | 結果 |\n")
                f.write("|------|------|\n")
                f.write(f"| 合格 | {success_count} |\n")
                f.write(f"| 不合格 | {failure_count} |\n")
                f.write(f"| スキップ | {skipped_count} |\n")
                f.write(f"| 合計 | {total_count} |\n\n")
                
                # 実行時間統計
                f.write("## 実行時間統計\n\n")
                successful_durations = [r.duration for r in self.results if r.success is True]
                failed_durations = [r.duration for r in self.results if r.success is False]
                
                if successful_durations:
                    avg_success_time = sum(successful_durations) / len(successful_durations)
                    max_success_time = max(successful_durations)
                    min_success_time = min(successful_durations)
                    
                    f.write("### 成功テスト実行時間統計\n\n")
                    f.write("| 統計 | 時間(秒) |\n")
                    f.write("|------|------|\n")
                    f.write(f"| 平均実行時間 | {avg_success_time:.2f} |\n")
                    f.write(f"| 最長実行時間 | {max_success_time:.2f} |\n")
                    f.write(f"| 最短実行時間 | {min_success_time:.2f} |\n\n")
                
                if failed_durations:
                    avg_fail_time = sum(failed_durations) / len(failed_durations)
                    max_fail_time = max(failed_durations)
                    min_fail_time = min(failed_durations)
                    
                    f.write("### 失敗テスト実行時間統計\n\n")
                    f.write("| 統計 | 時間(秒) |\n")
                    f.write("|------|------|\n")
                    f.write(f"| 平均実行時間 | {avg_fail_time:.2f} |\n")
                    f.write(f"| 最長実行時間 | {max_fail_time:.2f} |\n")
                    f.write(f"| 最短実行時間 | {min_fail_time:.2f} |\n\n")
                
                # カテゴリ別結果
                f.write("## カテゴリ別結果\n\n")
                categories = {}
                for result in self.results:
                    if result.category not in categories:
                        categories[result.category] = {
                            'success': 0,
                            'failure': 0,
                            'skipped': 0,
                            'total': 0
                        }
                    
                    categories[result.category]['total'] += 1
                    if result.success is True:
                        categories[result.category]['success'] += 1
                    elif result.success is False:
                        categories[result.category]['failure'] += 1
                    else:
                        categories[result.category]['skipped'] += 1
                
                f.write("| カテゴリ | 合格 | 不合格 | スキップ | 合計 | 成功率 |\n")
                f.write("|----------|------|--------|----------|------|--------|\n")
                
                for category, stats in sorted(categories.items()):
                    success_rate = 0
                    if stats['total'] > 0 and (stats['success'] + stats['failure']) > 0:
                        success_rate = (stats['success'] / (stats['success'] + stats['failure'])) * 100
                    
                    f.write(f"| {category} | {stats['success']} | {stats['failure']} | {stats['skipped']} | {stats['total']} | {success_rate:.1f}% |\n")
                
                f.write("\n")
                
                # 優先度別結果
                priority_stats = self._get_priority_stats()
                
                if len(priority_stats) > 1:  # 複数の優先度がある場合のみ表示
                    f.write("## 優先度別結果\n\n")
                    f.write("| 優先度 | 合格 | 不合格 | スキップ | 合計 | 成功率 |\n")
                    f.write("|--------|------|--------|----------|------|--------|\n")
                    
                    for priority, stats in sorted(priority_stats.items()):
                        success_rate = 0
                        if stats['total'] > 0 and (stats['success'] + stats['failure']) > 0:
                            success_rate = (stats['success'] / (stats['success'] + stats['failure'])) * 100
                        
                        f.write(f"| {priority} | {stats['success']} | {stats['failure']} | {stats['skipped']} | {stats['total']} | {success_rate:.1f}% |\n")
                    
                    f.write("\n")
                
                # 失敗したテストの一覧
                failed_tests = [r for r in self.results if r.success is False]
                if failed_tests:
                    f.write("## 不合格テスト一覧\n\n")
                    f.write("| テストID | カテゴリ | テスト名 | 失敗理由 |\n")
                    f.write("|----------|----------|----------|----------|\n")
                    
                    for result in failed_tests:
                        # メッセージが長い場合は省略
                        message = result.message
                        if len(message) > 100:
                            message = message[:97] + "..."
                        
                        # テーブルセル内の改行を置換
                        message = message.replace("\n", " ")
                        
                        f.write(f"| {result.test_id} | {result.category} | {result.name} | {message} |\n")
                    
                    f.write("\n")
                
                # 詳細結果
                f.write("## テスト詳細結果\n\n")
                
                for result in self.results:
                    status = "⭕ 合格" if result.success is True else "❌ 不合格" if result.success is False else "⏩ スキップ"
                    f.write(f"### {result.test_id}: {result.name} ({status})\n\n")
                    f.write(f"- **カテゴリ**: {result.category}\n")
                    f.write(f"- **実行時間**: {result.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"- **所要時間**: {result.duration:.2f}秒\n")
                    f.write(f"- **結果**: {result.message}\n\n")
                    
                    # 依存関係情報を表示
                    if result.test_id in self.test_dependencies:
                        f.write("#### 依存テスト結果\n\n")
                        f.write("| 依存テストID | 結果 |\n")
                        f.write("|--------------|------|\n")
                        
                        for dep in self.test_dependencies[result.test_id]:
                            status_text = "成功" if dep['satisfied'] else "失敗"
                            f.write(f"| {dep['dependency_id']} | {status_text} |\n")
                        
                        f.write("\n")
                    
                    f.write("#### 詳細\n\n")
                    for detail in result.details:
                        f.write(f"{detail}\n\n")
                    
                    if result.error:
                        f.write("#### エラー詳細\n\n")
                        f.write(f"```\n{str(result.error)}\n```\n\n")
                    
                    f.write("---\n\n")
                
                logger.info(f"テスト結果レポートを生成しました: {output_file}")
        except Exception as e:
            logger.error(f"レポート生成中にエラーが発生: {e}")
    
    def generate_html_report(self, output_file):
        """テスト結果のレポートをHTML形式で生成"""
        try:
            # HTMLレポートのテンプレート
            html_template = """<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Proxmoxクラスターテスト結果レポート</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 20px;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .success {
            color: green;
        }
        .failure {
            color: red;
        }
        .skipped {
            color: orange;
        }
        .summary-box {
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 15px;
            margin-bottom: 20px;
            display: inline-block;
            margin-right: 15px;
            min-width: 120px;
            text-align: center;
        }
        .summary-box h3 {
            margin-top: 0;
        }
        .success-bg {
            background-color: #d4edda;
        }
        .failure-bg {
            background-color: #f8d7da;
        }
        .skipped-bg {
            background-color: #fff3cd;
        }
        .detail-box {
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 15px;
            margin-bottom: 20px;
        }
        .detail-content {
            white-space: pre-wrap;
            font-family: monospace;
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
        }
        .error-box {
            background-color: #f8d7da;
            padding: 10px;
            border-radius: 4px;
            margin-top: 10px;
        }
        .tabs {
            overflow: hidden;
            border: 1px solid #ccc;
            background-color: #f1f1f1;
            border-radius: 4px 4px 0 0;
        }
        .tab {
            background-color: inherit;
            float: left;
            border: none;
            outline: none;
            cursor: pointer;
            padding: 12px 16px;
            transition: 0.3s;
            font-size: 17px;
        }
        .tab:hover {
            background-color: #ddd;
        }
        .tab.active {
            background-color: #ccc;
        }
        .tabcontent {
            display: none;
            padding: 6px 12px;
            border: 1px solid #ccc;
            border-top: none;
            border-radius: 0 0 4px 4px;
            animation: fadeEffect 1s;
        }
        @keyframes fadeEffect {
            from {opacity: 0;}
            to {opacity: 1;}
        }
        .chart-container {
            width: 100%;
            max-width: 600px;
            margin: 20px auto;
            height: 300px;
        }
        .progress-bar {
            height: 20px;
            background-color: #e9ecef;
            border-radius: 5px;
            margin-bottom: 10px;
            overflow: hidden;
        }
        .progress {
            height: 100%;
            line-height: 20px;
            color: white;
            text-align: center;
            background-color: #4CAF50;
        }
        .test-filter {
            margin-bottom: 15px;
        }
        .test-filter input {
            padding: 8px;
            margin-right: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            width: 300px;
        }
        .test-filter button {
            padding: 8px 12px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        .test-filter button:hover {
            background-color: #45a049;
        }
        .collapsible {
            background-color: #f1f1f1;
            color: #444;
            cursor: pointer;
            padding: 18px;
            width: 100%;
            border: none;
            text-align: left;
            outline: none;
            font-size: 15px;
            margin-bottom: 5px;
            border-radius: 4px;
        }
        .active-collapse, .collapsible:hover {
            background-color: #ddd;
        }
        .content {
            padding: 0 18px;
            display: none;
            overflow: hidden;
            background-color: #f9f9f9;
            border-radius: 0 0 4px 4px;
        }
    </style>
</head>
<body>
    <h1>Proxmoxクラスターテスト結果レポート</h1>
    <p>実行日時: {timestamp}</p>

    <div class="tabs">
        <button class="tab active" onclick="openTab(event, 'Summary')">サマリー</button>
        <button class="tab" onclick="openTab(event, 'CategoryResults')">カテゴリ別結果</button>
        <button class="tab" onclick="openTab(event, 'FailedTests')">不合格テスト</button>
        <button class="tab" onclick="openTab(event, 'DetailedResults')">詳細結果</button>
    </div>

    <div id="Summary" class="tabcontent" style="display: block;">
        <h2>テスト実行サマリー</h2>
        <div class="summary-box success-bg">
            <h3>合格</h3>
            <p class="success">{success_count}</p>
        </div>
        <div class="summary-box failure-bg">
            <h3>不合格</h3>
            <p class="failure">{failure_count}</p>
        </div>
        <div class="summary-box skipped-bg">
            <h3>スキップ</h3>
            <p class="skipped">{skipped_count}</p>
        </div>
        <div class="summary-box">
            <h3>合計</h3>
            <p>{total_count}</p>
        </div>
        <div style="clear: both;"></div>
        
        <h3>テスト成功率</h3>
        <div class="progress-bar">
            <div class="progress" style="width: {success_rate}%;">{success_rate}%</div>
        </div>

        <div class="chart-container">
            <canvas id="resultChart"></canvas>
        </div>

        {execution_time_stats}
    </div>

    <div id="CategoryResults" class="tabcontent">
        <h2>カテゴリ別結果</h2>
        <div class="chart-container">
            <canvas id="categoryChart"></canvas>
        </div>
        <table>
            <tr>
                <th>カテゴリ</th>
                <th>合格</th>
                <th>不合格</th>
                <th>スキップ</th>
                <th>合計</th>
                <th>成功率</th>
            </tr>
            {category_rows}
        </table>

        {priority_stats}
    </div>

    <div id="FailedTests" class="tabcontent">
        <h2>不合格テスト一覧</h2>
        
        <div class="test-filter">
            <input type="text" id="failedTestFilter" placeholder="テストIDまたは名前で検索...">
            <button onclick="filterFailedTests()">検索</button>
            <button onclick="clearFailedTestFilter()">クリア</button>
        </div>
        
        {failed_tests_table}
    </div>

    <div id="DetailedResults" class="tabcontent">
        <h2>テスト詳細結果</h2>
        
        <div class="test-filter">
            <input type="text" id="testFilter" placeholder="テストIDまたは名前で検索...">
            <button onclick="filterTests()">検索</button>
            <button onclick="clearTestFilter()">クリア</button>
        </div>
        
        {detailed_results}
    </div>

    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
    // タブ切り替え
    function openTab(evt, tabName) {
        var i, tabcontent, tablinks;
        tabcontent = document.getElementsByClassName("tabcontent");
        for (i = 0; i < tabcontent.length; i++) {
            tabcontent[i].style.display = "none";
        }
        tablinks = document.getElementsByClassName("tab");
        for (i = 0; i < tablinks.length; i++) {
            tablinks[i].className = tablinks[i].className.replace(" active", "");
        }
        document.getElementById(tabName).style.display = "block";
        evt.currentTarget.className += " active";
    }
    
    // 結果円グラフ
    var resultCtx = document.getElementById('resultChart').getContext('2d');
    var resultChart = new Chart(resultCtx, {
        type: 'pie',
        data: {
            labels: ['合格', '不合格', 'スキップ'],
            datasets: [{
                data: [{success_count}, {failure_count}, {skipped_count}],
                backgroundColor: [
                    '#4CAF50', // 緑（成功）
                    '#F44336', // 赤（失敗）
                    '#FFC107'  // 黄（スキップ）
                ],
                borderColor: '#fff',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'right',
                }
            }
        }
    });
    
    // カテゴリ別グラフ
    var categoryCtx = document.getElementById('categoryChart').getContext('2d');
    var categoryChart = new Chart(categoryCtx, {
        type: 'bar',
        data: {
            labels: [{category_names}],
            datasets: [
                {
                    label: '合格',
                    data: [{category_success}],
                    backgroundColor: '#4CAF50'
                },
                {
                    label: '不合格',
                    data: [{category_failure}],
                    backgroundColor: '#F44336'
                },
                {
                    label: 'スキップ',
                    data: [{category_skipped}],
                    backgroundColor: '#FFC107'
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                x: {
                    stacked: true,
                },
                y: {
                    stacked: true,
                    beginAtZero: true
                }
            }
        }
    });
    
    // 不合格テストのフィルタリング
    function filterFailedTests() {
        var input = document.getElementById('failedTestFilter');
        var filter = input.value.toUpperCase();
        var table = document.getElementById('failedTestsTable');
        var tr = table.getElementsByTagName('tr');
        
        for (var i = 1; i < tr.length; i++) {
            var id = tr[i].getElementsByTagName('td')[0].textContent;
            var name = tr[i].getElementsByTagName('td')[2].textContent;
            if (id.toUpperCase().indexOf(filter) > -1 || name.toUpperCase().indexOf(filter) > -1) {
                tr[i].style.display = '';
            } else {
                tr[i].style.display = 'none';
            }
        }
    }
    
    function clearFailedTestFilter() {
        document.getElementById('failedTestFilter').value = '';
        var table = document.getElementById('failedTestsTable');
        var tr = table.getElementsByTagName('tr');
        for (var i = 1; i < tr.length; i++) {
            tr[i].style.display = '';
        }
    }
    
    // 詳細テスト結果のフィルタリング
    function filterTests() {
        var input = document.getElementById('testFilter');
        var filter = input.value.toUpperCase();
        var details = document.getElementsByClassName('detail-box');
        
        for (var i = 0; i < details.length; i++) {
            var id = details[i].getAttribute('data-id');
            var name = details[i].getAttribute('data-name');
            if (id.toUpperCase().indexOf(filter) > -1 || name.toUpperCase().indexOf(filter) > -1) {
                details[i].style.display = '';
            } else {
                details[i].style.display = 'none';
            }
        }
    }
    
    function clearTestFilter() {
        document.getElementById('testFilter').value = '';
        var details = document.getElementsByClassName('detail-box');
        for (var i = 0; i < details.length; i++) {
            details[i].style.display = '';
        }
    }
    
    // 詳細の折りたたみ機能
    var coll = document.getElementsByClassName("collapsible");
    for (var i = 0; i < coll.length; i++) {
        coll[i].addEventListener("click", function() {
            this.classList.toggle("active-collapse");
            var content = this.nextElementSibling;
            if (content.style.display === "block") {
                content.style.display = "none";
            } else {
                content.style.display = "block";
            }
        });
    }
    </script>
</body>
</html>
"""
            # サマリー情報の作成
            success_count = sum(1 for r in self.results if r.success is True)
            failure_count = sum(1 for r in self.results if r.success is False)
            skipped_count = sum(1 for r in self.results if r.success is None)
            total_count = len(self.results)
            
            # 成功率の計算
            success_rate = 0
            if total_count > 0 and (success_count + failure_count) > 0:
                success_rate = (success_count / (success_count + failure_count)) * 100
            
            # 実行時間統計の作成
            execution_time_stats = ""
            successful_durations = [r.duration for r in self.results if r.success is True]
            failed_durations = [r.duration for r in self.results if r.success is False]
            
            if successful_durations:
                avg_success_time = sum(successful_durations) / len(successful_durations)
                max_success_time = max(successful_durations)
                min_success_time = min(successful_durations)
                
                execution_time_stats += """
                <h3>成功テスト実行時間統計</h3>
                <table>
                    <tr>
                        <th>統計</th>
                        <th>時間(秒)</th>
                    </tr>
                    <tr>
                        <td>平均実行時間</td>
                        <td>{:.2f}</td>
                    </tr>
                    <tr>
                        <td>最長実行時間</td>
                        <td>{:.2f}</td>
                    </tr>
                    <tr>
                        <td>最短実行時間</td>
                        <td>{:.2f}</td>
                    </tr>
                </table>
                """.format(avg_success_time, max_success_time, min_success_time)
            
            if failed_durations:
                avg_fail_time = sum(failed_durations) / len(failed_durations)
                max_fail_time = max(failed_durations)
                min_fail_time = min(failed_durations)
                
                execution_time_stats += """
                <h3>失敗テスト実行時間統計</h3>
                <table>
                    <tr>
                        <th>統計</th>
                        <th>時間(秒)</th>
                    </tr>
                    <tr>
                        <td>平均実行時間</td>
                        <td>{:.2f}</td>
                    </tr>
                    <tr>
                        <td>最長実行時間</td>
                        <td>{:.2f}</td>
                    </tr>
                    <tr>
                        <td>最短実行時間</td>
                        <td>{:.2f}</td>
                    </tr>
                </table>
                """.format(avg_fail_time, max_fail_time, min_fail_time)
            
            # カテゴリ別結果
            categories = {}
            for result in self.results:
                if result.category not in categories:
                    categories[result.category] = {
                        'success': 0,
                        'failure': 0,
                        'skipped': 0,
                        'total': 0
                    }
                
                categories[result.category]['total'] += 1
                if result.success is True:
                    categories[result.category]['success'] += 1
                elif result.success is False:
                    categories[result.category]['failure'] += 1
                else:
                    categories[result.category]['skipped'] += 1
            
            # カテゴリテーブルの行
            category_rows = ""
            category_names = []
            category_success = []
            category_failure = []
            category_skipped = []
            
            for category, stats in sorted(categories.items()):
                success_rate = 0
                if stats['total'] > 0 and (stats['success'] + stats['failure']) > 0:
                    success_rate = (stats['success'] / (stats['success'] + stats['failure'])) * 100
                
                category_rows += f"""
                <tr>
                    <td>{category}</td>
                    <td class="success">{stats['success']}</td>
                    <td class="failure">{stats['failure']}</td>
                    <td class="skipped">{stats['skipped']}</td>
                    <td>{stats['total']}</td>
                    <td>{success_rate:.1f}%</td>
                </tr>
                """
                
                # グラフ用データ
                category_names.append(f"'{category}'")
                category_success.append(str(stats['success']))
                category_failure.append(str(stats['failure']))
                category_skipped.append(str(stats['skipped']))
            
            # 優先度別結果
            priority_stats = ""
# 優先度別結果
            priority_stats_dict = self._get_priority_stats()
            
            if len(priority_stats_dict) > 1:  # 複数の優先度がある場合のみ表示
                priority_stats = """
                <h3>優先度別結果</h3>
                <table>
                    <tr>
                        <th>優先度</th>
                        <th>合格</th>
                        <th>不合格</th>
                        <th>スキップ</th>
                        <th>合計</th>
                        <th>成功率</th>
                    </tr>
                """
                
                for priority, stats in sorted(priority_stats_dict.items()):
                    success_rate = 0
                    if stats['total'] > 0 and (stats['success'] + stats['failure']) > 0:
                        success_rate = (stats['success'] / (stats['success'] + stats['failure'])) * 100
                    
                    priority_stats += f"""
                    <tr>
                        <td>{priority}</td>
                        <td class="success">{stats['success']}</td>
                        <td class="failure">{stats['failure']}</td>
                        <td class="skipped">{stats['skipped']}</td>
                        <td>{stats['total']}</td>
                        <td>{success_rate:.1f}%</td>
                    </tr>
                    """
                
                priority_stats += "</table>"
            
            # 失敗したテストの一覧
            failed_tests = [r for r in self.results if r.success is False]
            failed_tests_table = ""
            
            if failed_tests:
                failed_tests_table = """
                <table id="failedTestsTable">
                    <tr>
                        <th>テストID</th>
                        <th>カテゴリ</th>
                        <th>テスト名</th>
                        <th>失敗理由</th>
                    </tr>
                """
                
                for result in failed_tests:
                    # メッセージが長い場合は省略
                    message = result.message
                    if len(message) > 100:
                        message = message[:97] + "..."
                    
                    # HTML特殊文字をエスケープ
                    message = message.replace("<", "&lt;").replace(">", "&gt;")
                    # テーブルセル内の改行を置換
                    message = message.replace("\n", " ")
                    
                    failed_tests_table += f"""
                    <tr>
                        <td>{result.test_id}</td>
                        <td>{result.category}</td>
                        <td>{result.name}</td>
                        <td>{message}</td>
                    </tr>
                    """
                
                failed_tests_table += "</table>"
            else:
                failed_tests_table = "<p>不合格のテストはありません</p>"
            
            # 詳細結果
            detailed_results = ""
            
            for result in self.results:
                status_class = "success" if result.success is True else "failure" if result.success is False else "skipped"
                status_text = "合格" if result.success is True else "不合格" if result.success is False else "スキップ"
                
                # 詳細ボックス
                detailed_results += f"""
                <div class="detail-box" data-id="{result.test_id}" data-name="{result.name}">
                    <button class="collapsible">
                        {result.test_id}: {result.name} 
                        <span class="{status_class}" style="float: right;">{status_text}</span>
                    </button>
                    <div class="content">
                        <p><strong>カテゴリ:</strong> {result.category}</p>
                        <p><strong>実行時間:</strong> {result.start_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                        <p><strong>所要時間:</strong> {result.duration:.2f}秒</p>
                        <p><strong>結果:</strong> {result.message}</p>
                """
                
                # 依存関係情報を表示
                if result.test_id in self.test_dependencies:
                    detailed_results += """
                    <h4>依存テスト結果</h4>
                    <table>
                        <tr>
                            <th>依存テストID</th>
                            <th>結果</th>
                        </tr>
                    """
                    
                    for dep in self.test_dependencies[result.test_id]:
                        status_class = "success" if dep['satisfied'] else "failure"
                        status_text = "成功" if dep['satisfied'] else "失敗"
                        
                        detailed_results += f"""
                        <tr>
                            <td>{dep['dependency_id']}</td>
                            <td class="{status_class}">{status_text}</td>
                        </tr>
                        """
                    
                    detailed_results += "</table>"
                
                # 詳細情報
                if result.details:
                    detailed_results += "<h4>詳細</h4>"
                    details_text = "\n".join(result.details)
                    # HTML特殊文字をエスケープ
                    details_text = details_text.replace("<", "&lt;").replace(">", "&gt;")
                    detailed_results += f'<div class="detail-content">{details_text}</div>'
                
                # エラー詳細
                if result.error:
                    detailed_results += "<h4>エラー詳細</h4>"
                    error_text = str(result.error)
                    # HTML特殊文字をエスケープ
                    error_text = error_text.replace("<", "&lt;").replace(">", "&gt;")
                    detailed_results += f'<div class="error-box">{error_text}</div>'
                
                detailed_results += """
                    </div>
                </div>
                """
            
            # テンプレートの変数を置換
            html_content = html_template.format(
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                success_count=success_count,
                failure_count=failure_count,
                skipped_count=skipped_count,
                total_count=total_count,
                success_rate=round(success_rate, 1),
                execution_time_stats=execution_time_stats,
                category_rows=category_rows,
                category_names=", ".join(category_names),
                category_success=", ".join(category_success),
                category_failure=", ".join(category_failure),
                category_skipped=", ".join(category_skipped),
                priority_stats=priority_stats,
                failed_tests_table=failed_tests_table,
                detailed_results=detailed_results
            )
            
            # HTMLファイルを書き込み
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"HTML形式のテスト結果レポートを生成しました: {output_file}")
        except Exception as e:
            logger.error(f"HTMLレポート生成中にエラーが発生: {e}")
    
    def generate_junit_report(self, output_file):
        """テスト結果のレポートをJUnit XML形式で生成"""
        try:
            import xml.etree.ElementTree as ET
            from xml.dom import minidom
            
            # JUnitXML形式のルート要素
            test_suites = ET.Element("testsuites")
            test_suites.set("name", "ProxmoxClusterTests")
            test_suites.set("time", str(sum(r.duration for r in self.results)))
            test_suites.set("tests", str(len(self.results)))
            test_suites.set("failures", str(sum(1 for r in self.results if r.success is False)))
            test_suites.set("skipped", str(sum(1 for r in self.results if r.success is None)))
            
            # カテゴリごとにテストスイートを作成
            categories = {}
            for result in self.results:
                if result.category not in categories:
                    categories[result.category] = []
                categories[result.category].append(result)
            
            for category, results in sorted(categories.items()):
                test_suite = ET.SubElement(test_suites, "testsuite")
                test_suite.set("name", category)
                test_suite.set("tests", str(len(results)))
                test_suite.set("failures", str(sum(1 for r in results if r.success is False)))
                test_suite.set("skipped", str(sum(1 for r in results if r.success is None)))
                test_suite.set("time", str(sum(r.duration for r in results)))
                test_suite.set("timestamp", datetime.now().isoformat())
                
                for result in results:
                    test_case = ET.SubElement(test_suite, "testcase")
                    test_case.set("name", result.name)
                    test_case.set("classname", result.test_id)
                    test_case.set("time", str(result.duration))
                    
                    if result.success is False:
                        failure = ET.SubElement(test_case, "failure")
                        failure.set("message", result.message)
                        failure.set("type", "AssertionError")
                        
                        # 詳細情報を結合
                        failure_text = "\n".join(result.details)
                        if result.error:
                            failure_text += f"\n\nError: {str(result.error)}"
                        
                        failure.text = failure_text
                    
                    elif result.success is None:
                        skipped = ET.SubElement(test_case, "skipped")
                        skipped.set("message", result.message or "テストがスキップされました")
                    
                    # 出力ログ
                    system_out = ET.SubElement(test_case, "system-out")
                    system_out.text = "\n".join(result.details)
                    
                    # エラーログ
                    if result.error:
                        system_err = ET.SubElement(test_case, "system-err")
                        system_err.text = str(result.error)
            
            # XMLを整形して出力
            rough_string = ET.tostring(test_suites, "utf-8")
            reparsed = minidom.parseString(rough_string)
            pretty_xml = reparsed.toprettyxml(indent="  ")
            
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(pretty_xml)
            
            logger.info(f"JUnit XMLレポートを生成しました: {output_file}")
        except Exception as e:
            logger.error(f"JUnit XMLレポート生成中にエラーが発生: {e}")
    
    def _get_priority_stats(self):
        """優先度ごとの統計情報を取得する補助メソッド"""
        priority_stats = {}
        
        for result in self.results:
            # 優先度情報を取得（テスト結果オブジェクトに優先度属性が追加されている前提）
            # 実際の実装では、優先度の情報をどこかから取得する必要がある
            # 例えば、テストケース情報から取得するなど
            priority = getattr(result, 'priority', 'unknown')
            
            if priority not in priority_stats:
                priority_stats[priority] = {
                    'success': 0,
                    'failure': 0,
                    'skipped': 0,
                    'total': 0
                }
            
            priority_stats[priority]['total'] += 1
            if result.success is True:
                priority_stats[priority]['success'] += 1
            elif result.success is False:
                priority_stats[priority]['failure'] += 1
            else:
                priority_stats[priority]['skipped'] += 1
        
        return priority_stats
    
    def generate_csv_report(self, output_file):
        """テスト結果をCSV形式で出力"""
        try:
            import csv
            
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # ヘッダー行
                writer.writerow([
                    'テストID', 'カテゴリ', 'テスト名', '結果', 'メッセージ', 
                    '開始時間', '終了時間', '所要時間(秒)', 'エラー'
                ])
                
                # データ行
                for result in self.results:
                    status = "合格" if result.success is True else "不合格" if result.success is False else "スキップ"
                    error_text = str(result.error) if result.error else ""
                    
                    writer.writerow([
                        result.test_id,
                        result.category,
                        result.name,
                        status,
                        result.message,
                        result.start_time.strftime('%Y-%m-%d %H:%M:%S'),
                        result.end_time.strftime('%Y-%m-%d %H:%M:%S') if result.end_time else "",
                        f"{result.duration:.2f}",
                        error_text
                    ])
            
            logger.info(f"CSV形式のテスト結果レポートを生成しました: {output_file}")
        except Exception as e:
            logger.error(f"CSVレポート生成中にエラーが発生: {e}")
    
    def generate_dependency_graph(self, output_file):
        """テスト依存関係のグラフを生成 (Graphviz DOT形式)"""
        try:
            # DOTファイルのヘッダー
            dot_content = "digraph TestDependencies {\n"
            dot_content += "  rankdir=LR;\n"  # 左から右へのレイアウト
            dot_content += "  node [shape=box, style=\"rounded,filled\", fontname=\"Helvetica\"];\n"
            
            # ノード（テスト）の定義
            for result in self.results:
                color = "green" if result.success is True else "red" if result.success is False else "yellow"
                dot_content += f'  "{result.test_id}" [label="{result.test_id}\\n{result.name}", fillcolor="{color}"];\n'
            
            # エッジ（依存関係）の定義
            for test_id, dependencies in self.test_dependencies.items():
                for dep in dependencies:
                    dep_id = dep['dependency_id']
                    color = "green" if dep['satisfied'] else "red"
                    dot_content += f'  "{dep_id}" -> "{test_id}" [color="{color}"];\n'
            
            dot_content += "}\n"
            
            # DOTファイルを書き込み
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(dot_content)
            
            logger.info(f"テスト依存関係グラフを生成しました: {output_file}")
            
            # PNGへの変換（オプション - Graphvizがインストールされている場合）
            try:
                import subprocess
                png_file = output_file.rsplit('.', 1)[0] + '.png'
                subprocess.run(['dot', '-Tpng', output_file, '-o', png_file], check=True)
                logger.info(f"依存関係グラフのPNG画像を生成しました: {png_file}")
            except Exception as e:
                logger.warning(f"PNGへの変換に失敗しました（Graphvizがインストールされていない可能性があります）: {e}")
                
        except Exception as e:
            logger.error(f"依存関係グラフ生成中にエラーが発生: {e}")
    
    def generate_all_reports(self, base_filename):
        """すべての形式のレポートを生成"""
        base_path = os.path.splitext(base_filename)[0]
        
        # Markdown レポート
        self.generate_report(f"{base_path}.md")
        
        # HTML レポート
        self.generate_html_report(f"{base_path}.html")
        
        # JUnit XML レポート
        self.generate_junit_report(f"{base_path}.xml")
        
        # CSV レポート
        self.generate_csv_report(f"{base_path}.csv")
        
        # 依存関係グラフ
        if self.test_dependencies:
            self.generate_dependency_graph(f"{base_path}_dependencies.dot")
        
        logger.info(f"すべての形式のレポートを生成しました: {base_path}.*")

# TestResult クラスを拡張して、優先度情報を含めるように修正
class TestResult:
    """テスト結果クラス"""
    
    def __init__(self, test_id: str, category: str, name: str, priority: str = None, start_time: datetime = None):
        """
        初期化メソッド
        
        Args:
            test_id: テストケースID
            category: テストカテゴリ
            name: テスト名
            priority: テスト優先度
            start_time: 開始時刻（省略時は現在時刻）
        """
        self.test_id = test_id
        self.category = category
        self.name = name
        self.priority = priority
        self.start_time = start_time or datetime.now()
        self.end_time = None
        self.success = None  # None: 未実行, True: 成功, False: 失敗
        self.message = ""
        self.details = []
        self.error = None
    
    def set_success(self, message: str = "") -> None:
        """テスト成功を設定"""
        self.success = True
        self.message = message or "テスト成功"
        self.end_time = datetime.now()
    
    def set_failure(self, message: str = "", error: Exception = None) -> None:
        """テスト失敗を設定"""
        self.success = False
        self.message = message or "テスト失敗"
        self.error = error
        self.end_time = datetime.now()
    
    def add_detail(self, message: str) -> None:
        """詳細メッセージを追加"""
        self.details.append(message)
    
    @property
    def duration(self) -> float:
        """テスト実行時間（秒）"""
        if not self.end_time:
            return 0
        delta = self.end_time - self.start_time
        return delta.total_seconds()

# TestExecutor クラスの execute_test メソッドを修正して、TestResult オブジェクトに優先度情報を設定
def execute_test(self, test_case: Dict) -> TestResult:
    """
    テストを実行
    
    Args:
        test_case: テストケース辞書
        
    Returns:
        テスト結果
    """
    test_id = test_case.get('test_case_id', str(uuid.uuid4())[:8])
    category = test_case.get('category', 'uncategorized')
    test_name = test_case.get('purpose', test_id)
    test_method = test_case.get('test_method', '')
    node_key = test_case.get('node', '')
    command = test_case.get('command', '')
    validation_method = test_case.get('validation_method', '')
    
    # 優先度情報を取得
    priority = test_case.get('priority', 'unknown')
    
    # 結果オブジェクトを初期化（優先度情報を含める）
    result = TestResult(test_id, category, test_name, priority)
    
    # 残りのメソッド内容は同じ
    # ...
    
    return result

    
def generate_junit_report(self, output_file: str) -> None:
    """
    テスト結果のレポートをJUnit XML形式で生成
    
    Args:
        output_file: 出力ファイルパス
    """
    try:
        import xml.etree.ElementTree as ET
        from xml.dom import minidom
        
        # JUnitXML形式のルート要素
        test_suites = ET.Element("testsuites")
        
        # カテゴリごとにテストスイートを作成
        categories = {}
        for result in self.results:
            if result.category not in categories:
                categories[result.category] = []
            categories[result.category].append(result)
        
        for category, results in sorted(categories.items()):
            test_suite = ET.SubElement(test_suites, "testsuite")
            test_suite.set("name", category)
            test_suite.set("tests", str(len(results)))
            test_suite.set("failures", str(sum(1 for r in results if r.success is False)))
            test_suite.set("skipped", str(sum(1 for r in results if r.success is None)))
            test_suite.set("timestamp", datetime.now().isoformat())
            
            for result in results:
                test_case = ET.SubElement(test_suite, "testcase")
                test_case.set("name", result.name)
                test_case.set("classname", result.test_id)
                test_case.set("time", str(result.duration))
                
                if result.success is False:
                    failure = ET.SubElement(test_case, "failure")
                    failure.set("message", result.message)
                    failure.text = "\n".join(result.details)
                    if result.error:
                        failure.text += f"\n\nError: {str(result.error)}"
                
                elif result.success is None:
                    skipped = ET.SubElement(test_case, "skipped")
                    skipped.set("message", result.message or "テストがスキップされました")
                
                system_out = ET.SubElement(test_case, "system-out")
                system_out.text = "\n".join(result.details)
        
        # XMLを整形して出力
        rough_string = ET.tostring(test_suites, "utf-8")
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(pretty_xml)
        
        logger.info(f"JUnit XMLレポートを生成しました: {output_file}")
    except Exception as e:
        logger.error(f"JUnit XMLレポート生成中にエラーが発生: {e}")

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='Proxmox クラスターテスト自動化')
    parser.add_argument('--config', help='設定ファイルパス')
    parser.add_argument('--test-cases', help='テストケースファイルパス (.csv, .tsvなど)', required=True)
    parser.add_argument('--output', help='テスト結果レポート出力パス', default='proxmox_test_report.md')
    parser.add_argument('--junit', help='JUnit XML形式レポート出力パス')
    parser.add_argument('--category', help='実行するテストカテゴリ (カンマ区切りで複数指定可)')
    parser.add_argument('--priority', help='実行するテスト優先度 (カンマ区切りで複数指定可)')
    parser.add_argument('--id', help='特定のテストIDを実行 (カンマ区切りで複数指定可)')
    parser.add_argument('--skip-manual', action='store_true', help='手動テストをスキップする')
    args = parser.parse_args()
    
    try:
        # テスターを初期化
        tester = ProxmoxTester(config_file=args.config)
        
        # テストケースを読み込み
        test_cases = tester.load_test_cases(args.test_cases)
        
        if not test_cases:
            logger.error(f"No test cases found in {args.test_cases}")
            return 1
        
        # フィルタリング条件の解析
        categories = []
        if args.category:
            categories = [cat.strip() for cat in args.category.split(',')]
        
        priorities = []
        if args.priority:
            priorities = [pri.strip() for pri in args.priority.split(',')]
        
        test_ids = []
        if args.id:
            test_ids = [id.strip() for id in args.id.split(',')]
        
        # テストケースをフィルタリング
        filtered_test_cases = []
        for test_case in test_cases:
            # カテゴリでフィルタリング
            if categories and test_case.get('category') not in categories:
                continue
                
            # 優先度でフィルタリング
            if priorities and test_case.get('priority') not in priorities:
                continue
                
            # テストIDでフィルタリング
            if test_ids and test_case.get('test_case_id') not in test_ids:
                continue
                
            # 手動テストをスキップ
            if args.skip_manual and test_case.get('test_method', '').lower() == '手動':
                logger.info(f"Skipping manual test: {test_case.get('test_case_id')}")
                continue
                
            filtered_test_cases.append(test_case)
        
        logger.info(f"Filtered {len(filtered_test_cases)} of {len(test_cases)} test cases")
        
        # テストを実行
        for test_case in filtered_test_cases:
            tester.run_test_case(test_case)
        
        # レポートを生成
        tester.generate_report(args.output)
        
        # JUnit XMLレポートを生成（指定された場合）
        if args.junit:
            tester.generate_junit_report(args.junit)
        
        # 結果を表示
        success_count = sum(1 for r in tester.results if r.success is True)
        failure_count = sum(1 for r in tester.results if r.success is False)
        skipped_count = sum(1 for r in tester.results if r.success is None)
        
        logger.info(f"テスト完了: 合格={success_count}, 不合格={failure_count}, スキップ={skipped_count}, 合計={len(tester.results)}")
        logger.info(f"レポート出力: {args.output}")
        
        # 失敗が1つでもあればエラー終了
        return 0 if failure_count == 0 else 1
    
    except Exception as e:
        logger.exception(f"Error running tests: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
